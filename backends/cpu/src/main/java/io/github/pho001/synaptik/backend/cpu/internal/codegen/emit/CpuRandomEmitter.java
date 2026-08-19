package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;

/**
 * Emits direct typed explicit-state initializer and dropout bodies into generated CPU classes.
 * Family, represented type, carrier form, access regime, probability, initializer words, and
 * counter policies are resolved while the class is generated. Invocation state is primitive and
 * local. Dense arrays use integer loops; general layouts and segments use long addressing and
 * predefined native-order unaligned layouts.
 */
public final class CpuRandomEmitter {
    private static final ClassDesc SEGMENT = ClassDesc.of("java.lang.foreign.MemorySegment");
    private static final ClassDesc VALUE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout");
    private static final ClassDesc DOUBLE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfDouble");
    private static final ClassDesc FLOAT_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfFloat");
    private static final ClassDesc LONG_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfLong");
    private static final ClassDesc BYTE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout$OfByte");

    /** Creates a stateless emitter with no generator instance or run state. */
    public CpuRandomEmitter() { }

    /**
     * Emits one exact typed initializer or dropout body.
     * @param code non-null generated method body
     * @param specialization non-null scalar carrier specialization
     * @param ir non-null instruction-free random structural encoding
     * @throws NullPointerException if a required reference is null
     * @throws IllegalArgumentException if the specialization or random identity is inconsistent
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        Encoding p = Encoding.parse(ir);
        int boundaries = p.family == CpuRandomIr.Family.INITIAL_STATE ? 1 : 5;
        if (specialization.carrierPattern().size() != boundaries
                || specialization.executionStrategy().compute()
                    != io.github.pho001.synaptik.backend.cpu.internal.prepare
                        .CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR
                || specialization.scratchParameter()) {
            throw new IllegalArgumentException("random specialization facts disagree");
        }
        if (specialization.loopAddressing(ir)
                == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT) {
            emitDense(code, specialization, p);
        } else {
            emitGeneral(code, specialization, p);
        }
    }

    private static void emitDense(CodeBuilder code, CpuKernelSpecialization s, Encoding p) {
        int geometry = p.boundaryCount;
        int startParameter = geometry + 1;
        int endParameter = startParameter + 2;
        int start = code.allocateLocal(TypeKind.INT);
        int end = code.allocateLocal(TypeKind.INT);
        int[] bases = new int[p.boundaryCount];
        for (int boundary = 0; boundary < bases.length; boundary++) {
            bases[boundary] = code.allocateLocal(TypeKind.INT);
            geometry(code, geometry, p.layoutOffsets[boundary] + 1).l2i().istore(bases[boundary]);
        }
        code.lload(startParameter).l2i().istore(start).lload(endParameter).l2i().istore(end);
        var elementWork = code.newLabel();
        var complete = code.newLabel();
        code.iload(start).branch(Opcode.IFNE, elementWork);
        code.iload(end).branch(Opcode.IFNE, elementWork);
        if (p.family == CpuRandomIr.Family.INITIAL_STATE) {
            emitInitializer(code, s.carrierPattern().getFirst(), 0, bases[0], true,
                    p.keyBits, p.counterBits);
        } else {
            emitNextState(code, s, bases[1], bases[4], true, geometry);
        }
        code.branch(Opcode.GOTO, complete).labelBinding(elementWork);
        if (p.family == CpuRandomIr.Family.DROPOUT) {
            emitDenseDropout(code, s, p, start, end, bases);
        }
        code.labelBinding(complete);
    }

    private static void emitDenseDropout(CodeBuilder code, CpuKernelSpecialization s, Encoding p,
            int logical, int end, int[] bases) {
        int inputAddress = code.allocateLocal(TypeKind.INT);
        int outputAddress = code.allocateLocal(TypeKind.INT);
        int maskAddress = code.allocateLocal(TypeKind.INT);
        code.iload(bases[0]).iload(logical).iadd().istore(inputAddress);
        code.iload(bases[2]).iload(logical).iadd().istore(outputAddress);
        code.iload(bases[3]).iload(logical).iadd().istore(maskAddress);
        int key = code.allocateLocal(TypeKind.LONG);
        int counter = code.allocateLocal(TypeKind.LONG);
        int keyOffset = code.allocateLocal(TypeKind.LONG);
        int word = code.allocateLocal(TypeKind.LONG);
        int uniform = code.allocateLocal(TypeKind.DOUBLE);
        int denominator = code.allocateLocal(TypeKind.DOUBLE);
        int keep = code.allocateLocal(TypeKind.INT);
        int result = code.allocateLocal(p.valueType == DataType.FLOAT64
                ? TypeKind.DOUBLE : TypeKind.FLOAT);
        loadCarrier(code, DataType.INT64, s.carrierPattern().get(1), 1, bases[1], true);
        code.lstore(key);
        loadOffset(code, DataType.INT64, s.carrierPattern().get(1), 1, bases[1], 1, true);
        code.lstore(counter);
        code.lload(key).loadConstant(CpuRandomIr.KEY_BIAS).ladd().lstore(keyOffset);
        emitMix64(code, keyOffset);
        code.loadConstant(1.0d - Double.longBitsToDouble(p.probabilityBits)).dstore(denominator);
        var loop = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(loop).iload(logical).iload(end).branch(Opcode.IF_ICMPGE, done);
        code.lload(counter).iload(logical).i2l().ladd().lload(keyOffset).ladd().lstore(word);
        emitMix64(code, word);
        code.lload(word).loadConstant(11).lushr().l2d().loadConstant(0x1.0p-53)
                .dmul().dstore(uniform);
        emitKeep(code, uniform, Double.longBitsToDouble(p.probabilityBits), keep);
        storeCarrier(code, DataType.BOOL, s.carrierPattern().get(3), 3,
                maskAddress, keep, true);
        emitValue(code, s, p.valueType, inputAddress, outputAddress,
                denominator, keep, result, true);
        code.iinc(logical, 1).iinc(inputAddress, 1).iinc(outputAddress, 1)
                .iinc(maskAddress, 1).branch(Opcode.GOTO, loop).labelBinding(done);
    }

    private static void emitGeneral(CodeBuilder code, CpuKernelSpecialization s, Encoding p) {
        int geometry = p.boundaryCount;
        int startParameter = geometry + 1;
        int endParameter = startParameter + 2;
        LayoutLocals[] layouts = new LayoutLocals[p.boundaryCount];
        for (int boundary = 0; boundary < layouts.length; boundary++) {
            layouts[boundary] = loadLayout(code, geometry,
                    p.layoutOffsets[boundary], p.ranks[boundary]);
        }
        int start = code.allocateLocal(TypeKind.LONG);
        int end = code.allocateLocal(TypeKind.LONG);
        code.lload(startParameter).lstore(start).lload(endParameter).lstore(end);
        var elementWork = code.newLabel();
        var complete = code.newLabel();
        code.lload(start).lconst_0().lcmp().branch(Opcode.IFNE, elementWork);
        code.lload(end).lconst_0().lcmp().branch(Opcode.IFNE, elementWork);
        if (p.family == CpuRandomIr.Family.INITIAL_STATE) {
            emitInitializer(code, s.carrierPattern().getFirst(), 0, layouts[0].base,
                    layouts[0].strides[0], p.keyBits, p.counterBits);
        } else {
            emitNextState(code, s, layouts[1].base, layouts[1].strides[0],
                    layouts[4].base, layouts[4].strides[0], geometry);
        }
        code.branch(Opcode.GOTO, complete).labelBinding(elementWork);
        if (p.family == CpuRandomIr.Family.DROPOUT) {
            emitGeneralDropout(code, s, p, start, end, layouts);
        }
        code.labelBinding(complete);
    }

    private static void emitGeneralDropout(CodeBuilder code, CpuKernelSpecialization s, Encoding p,
            int logical, int end, LayoutLocals[] layouts) {
        int inputAddress = code.allocateLocal(TypeKind.LONG);
        int outputAddress = code.allocateLocal(TypeKind.LONG);
        int maskAddress = code.allocateLocal(TypeKind.LONG);
        int remainder = code.allocateLocal(TypeKind.LONG);
        int coordinate = code.allocateLocal(TypeKind.LONG);
        int key = code.allocateLocal(TypeKind.LONG);
        int counter = code.allocateLocal(TypeKind.LONG);
        int keyOffset = code.allocateLocal(TypeKind.LONG);
        int word = code.allocateLocal(TypeKind.LONG);
        int uniform = code.allocateLocal(TypeKind.DOUBLE);
        int denominator = code.allocateLocal(TypeKind.DOUBLE);
        int keep = code.allocateLocal(TypeKind.INT);
        int result = code.allocateLocal(p.valueType == DataType.FLOAT64
                ? TypeKind.DOUBLE : TypeKind.FLOAT);
        loadCarrier(code, DataType.INT64, s.carrierPattern().get(1), 1,
                layouts[1].base, false);
        code.lstore(key);
        int stateCounterAddress = code.allocateLocal(TypeKind.LONG);
        code.lload(layouts[1].base).lload(layouts[1].strides[0]).ladd()
                .lstore(stateCounterAddress);
        loadCarrier(code, DataType.INT64, s.carrierPattern().get(1), 1,
                stateCounterAddress, false);
        code.lstore(counter);
        code.lload(key).loadConstant(CpuRandomIr.KEY_BIAS).ladd().lstore(keyOffset);
        emitMix64(code, keyOffset);
        code.loadConstant(1.0d - Double.longBitsToDouble(p.probabilityBits)).dstore(denominator);
        var loop = code.newLabel();
        var done = code.newLabel();
        code.labelBinding(loop).lload(logical).lload(end).lcmp().branch(Opcode.IFGE, done);
        emitAddress(code, layouts[0], logical, inputAddress, remainder, coordinate);
        emitAddress(code, layouts[2], logical, outputAddress, remainder, coordinate);
        emitAddress(code, layouts[3], logical, maskAddress, remainder, coordinate);
        code.lload(counter).lload(logical).ladd().lload(keyOffset).ladd().lstore(word);
        emitMix64(code, word);
        code.lload(word).loadConstant(11).lushr().l2d().loadConstant(0x1.0p-53)
                .dmul().dstore(uniform);
        emitKeep(code, uniform, Double.longBitsToDouble(p.probabilityBits), keep);
        storeCarrier(code, DataType.BOOL, s.carrierPattern().get(3), 3,
                maskAddress, keep, false);
        emitValue(code, s, p.valueType, inputAddress, outputAddress,
                denominator, keep, result, false);
        code.lload(logical).lconst_1().ladd().lstore(logical)
                .branch(Opcode.GOTO, loop).labelBinding(done);
    }

    private static void emitInitializer(CodeBuilder code, CarrierAccess access, int parameter,
            int base, boolean ints, long keyBits, long counterBits) {
        int value = code.allocateLocal(TypeKind.LONG);
        code.loadConstant(keyBits).lstore(value);
        storeCarrier(code, DataType.INT64, access, parameter, base, value, ints);
        code.loadConstant(counterBits).lstore(value);
        storeOffset(code, DataType.INT64, access, parameter, base, 1, ints, value);
    }

    private static void emitInitializer(CodeBuilder code, CarrierAccess access, int parameter,
            int base, int stride, long keyBits, long counterBits) {
        int value = code.allocateLocal(TypeKind.LONG);
        int address = code.allocateLocal(TypeKind.LONG);
        code.loadConstant(keyBits).lstore(value);
        storeCarrier(code, DataType.INT64, access, parameter, base, value, false);
        code.lload(base).lload(stride).ladd().lstore(address);
        code.loadConstant(counterBits).lstore(value);
        storeCarrier(code, DataType.INT64, access, parameter, address, value, false);
    }

    private static void emitNextState(CodeBuilder code, CpuKernelSpecialization s,
            int stateBase, int nextBase, boolean ints, int geometry) {
        int key = code.allocateLocal(TypeKind.LONG);
        int counter = code.allocateLocal(TypeKind.LONG);
        loadCarrier(code, DataType.INT64, s.carrierPattern().get(1), 1, stateBase, ints);
        code.lstore(key);
        loadOffset(code, DataType.INT64, s.carrierPattern().get(1), 1, stateBase, 1, ints);
        code.lstore(counter);
        storeCarrier(code, DataType.INT64, s.carrierPattern().get(4), 4, nextBase, key, ints);
        code.lload(counter); geometry(code, geometry, 2).ladd().lstore(counter);
        storeOffset(code, DataType.INT64, s.carrierPattern().get(4), 4,
                nextBase, 1, ints, counter);
    }

    private static void emitNextState(CodeBuilder code, CpuKernelSpecialization s,
            int stateBase, int stateStride, int nextBase, int nextStride, int geometry) {
        int key = code.allocateLocal(TypeKind.LONG);
        int counter = code.allocateLocal(TypeKind.LONG);
        int address = code.allocateLocal(TypeKind.LONG);
        loadCarrier(code, DataType.INT64, s.carrierPattern().get(1), 1, stateBase, false);
        code.lstore(key);
        code.lload(stateBase).lload(stateStride).ladd().lstore(address);
        loadCarrier(code, DataType.INT64, s.carrierPattern().get(1), 1, address, false);
        code.lstore(counter);
        storeCarrier(code, DataType.INT64, s.carrierPattern().get(4), 4, nextBase, key, false);
        code.lload(nextBase).lload(nextStride).ladd().lstore(address);
        code.lload(counter); geometry(code, geometry, 2).ladd().lstore(counter);
        storeCarrier(code, DataType.INT64, s.carrierPattern().get(4), 4,
                address, counter, false);
    }

    private static void emitValue(CodeBuilder code, CpuKernelSpecialization s, DataType type,
            int inputAddress, int outputAddress, int denominator, int keep, int result,
            boolean ints) {
        var dropped = code.newLabel();
        var store = code.newLabel();
        code.iload(keep).branch(Opcode.IFEQ, dropped);
        loadCarrier(code, type, s.carrierPattern().get(0), 0, inputAddress, ints);
        if (type == DataType.FLOAT64) code.dload(denominator).ddiv().dstore(result);
        else code.f2d().dload(denominator).ddiv().d2f().fstore(result);
        code.branch(Opcode.GOTO, store).labelBinding(dropped);
        if (type == DataType.FLOAT64) code.loadConstant(0.0d).dstore(result);
        else code.loadConstant(0.0f).fstore(result);
        code.labelBinding(store);
        storeCarrier(code, type, s.carrierPattern().get(2), 2, outputAddress, result, ints);
    }

    private static void emitKeep(CodeBuilder code, int uniform, double probability, int keep) {
        var kept = code.newLabel();
        var selected = code.newLabel();
        code.dload(uniform).loadConstant(probability).dcmpl().branch(Opcode.IFGE, kept);
        code.loadConstant(0).istore(keep).branch(Opcode.GOTO, selected);
        code.labelBinding(kept).loadConstant(1).istore(keep).labelBinding(selected);
    }

    private static void emitMix64(CodeBuilder code, int value) {
        code.lload(value).lload(value).loadConstant(30).lushr().lxor()
                .loadConstant(CpuRandomIr.MIX_MULTIPLIER_1).lmul().lstore(value);
        code.lload(value).lload(value).loadConstant(27).lushr().lxor()
                .loadConstant(CpuRandomIr.MIX_MULTIPLIER_2).lmul().lstore(value);
        code.lload(value).lload(value).loadConstant(31).lushr().lxor().lstore(value);
    }

    private static LayoutLocals loadLayout(CodeBuilder code, int geometry, int offset, int rank) {
        int base = code.allocateLocal(TypeKind.LONG);
        int[] extents = new int[rank];
        int[] strides = new int[rank];
        geometry(code, geometry, offset + 1).lstore(base);
        for (int axis = 0; axis < rank; axis++) {
            extents[axis] = code.allocateLocal(TypeKind.LONG);
            geometry(code, geometry, offset + 2 + axis).lstore(extents[axis]);
            strides[axis] = code.allocateLocal(TypeKind.LONG);
            geometry(code, geometry, offset + 2 + rank + axis).lstore(strides[axis]);
        }
        return new LayoutLocals(base, extents, strides);
    }

    private static void emitAddress(CodeBuilder code, LayoutLocals layout, int logical,
            int address, int remainder, int coordinate) {
        code.lload(layout.base).lstore(address).lload(logical).lstore(remainder);
        for (int axis = layout.extents.length - 1; axis >= 0; axis--) {
            code.lload(remainder).lload(layout.extents[axis]).lrem().lstore(coordinate);
            code.lload(remainder).lload(layout.extents[axis]).ldiv().lstore(remainder);
            code.lload(address).lload(coordinate).lload(layout.strides[axis]).lmul()
                    .ladd().lstore(address);
        }
    }

    private static CodeBuilder geometry(CodeBuilder code, int parameter, int index) {
        return code.aload(parameter).loadConstant(index).laload();
    }

    private static void loadOffset(CodeBuilder code, DataType type, CarrierAccess access,
            int parameter, int base, int offset, boolean ints) {
        int address = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        if (ints) code.iload(base).loadConstant(offset).iadd().istore(address);
        else code.lload(base).loadConstant((long) offset).ladd().lstore(address);
        loadCarrier(code, type, access, parameter, address, ints);
    }

    private static void storeOffset(CodeBuilder code, DataType type, CarrierAccess access,
            int parameter, int base, int offset, boolean ints, int value) {
        int address = code.allocateLocal(ints ? TypeKind.INT : TypeKind.LONG);
        if (ints) code.iload(base).loadConstant(offset).iadd().istore(address);
        else code.lload(base).loadConstant((long) offset).ladd().lstore(address);
        storeCarrier(code, type, access, parameter, address, value, ints);
    }

    private static void loadCarrier(CodeBuilder code, DataType type, CarrierAccess access,
            int parameter, int address, boolean ints) {
        if (access != CarrierAccess.MEMORY_SEGMENT) {
            code.aload(parameter);
            if (ints) code.iload(address); else code.lload(address).l2i();
            switch (type) {
                case FLOAT64 -> code.daload(); case FLOAT32 -> code.faload();
                case INT64 -> code.laload();
                default -> throw new IllegalArgumentException("unsupported random load type");
            }
            return;
        }
        code.aload(parameter).getstatic(VALUE_LAYOUT, layoutField(type), layoutClass(type));
        byteOffset(code, type, address, ints);
        code.invokeinterface(SEGMENT, "get", MethodTypeDesc.of(primitive(type), layoutClass(type),
                ConstantDescs.CD_long));
    }

    private static void storeCarrier(CodeBuilder code, DataType type, CarrierAccess access,
            int parameter, int address, int value, boolean ints) {
        if (access != CarrierAccess.MEMORY_SEGMENT) {
            code.aload(parameter);
            if (ints) code.iload(address); else code.lload(address).l2i();
            loadValue(code, type, value);
            switch (type) {
                case FLOAT64 -> code.dastore(); case FLOAT32 -> code.fastore();
                case INT64 -> code.lastore(); case BOOL -> code.bastore();
                default -> throw new IllegalArgumentException("unsupported random store type");
            }
            return;
        }
        code.aload(parameter).getstatic(VALUE_LAYOUT, layoutField(type), layoutClass(type));
        byteOffset(code, type, address, ints);
        loadValue(code, type, value);
        code.invokeinterface(SEGMENT, "set", MethodTypeDesc.of(ConstantDescs.CD_void,
                layoutClass(type), ConstantDescs.CD_long, primitive(type)));
    }

    private static void byteOffset(CodeBuilder code, DataType type, int address, boolean ints) {
        if (ints) code.iload(address).i2l(); else code.lload(address);
        code.loadConstant((long) type.byteWidth()).lmul();
    }

    private static void loadValue(CodeBuilder code, DataType type, int value) {
        switch (type) {
            case FLOAT64 -> code.dload(value); case FLOAT32 -> code.fload(value);
            case INT64 -> code.lload(value); case BOOL -> code.iload(value);
            default -> throw new IllegalArgumentException("unsupported random value type");
        }
    }

    private static String layoutField(DataType type) {
        return switch (type) {
            case FLOAT64 -> "JAVA_DOUBLE_UNALIGNED"; case FLOAT32 -> "JAVA_FLOAT_UNALIGNED";
            case INT64 -> "JAVA_LONG_UNALIGNED"; case BOOL -> "JAVA_BYTE";
            default -> throw new IllegalArgumentException("unsupported random layout type");
        };
    }

    private static ClassDesc layoutClass(DataType type) {
        return switch (type) {
            case FLOAT64 -> DOUBLE_LAYOUT; case FLOAT32 -> FLOAT_LAYOUT;
            case INT64 -> LONG_LAYOUT; case BOOL -> BYTE_LAYOUT;
            default -> throw new IllegalArgumentException("unsupported random layout type");
        };
    }

    private static ClassDesc primitive(DataType type) {
        return switch (type) {
            case FLOAT64 -> ConstantDescs.CD_double; case FLOAT32 -> ConstantDescs.CD_float;
            case INT64 -> ConstantDescs.CD_long; case BOOL -> ConstantDescs.CD_byte;
            default -> throw new IllegalArgumentException("unsupported random primitive type");
        };
    }

    private record LayoutLocals(int base, int[] extents, int[] strides) { }

    private record Encoding(CpuRandomIr.Family family, DataType valueType, long keyBits,
            long counterBits, long probabilityBits, int boundaryCount, int[] ranks,
            int[] layoutOffsets) {
        private static Encoding parse(CpuKernelIr ir) {
            if (!ir.instructions().isEmpty() || !ir.familyIdentity().startsWith("random:")) {
                throw new IllegalArgumentException("encoded random identity is invalid");
            }
            String[] parts = ir.familyIdentity().split(":");
            CpuRandomIr.Family family = CpuRandomIr.Family.valueOf(parts[1]);
            long key = 0, counter = 0, probability = 0;
            for (String part : parts) {
                if (part.startsWith("key=")) key = Long.parseUnsignedLong(part.substring(4), 16);
                else if (part.startsWith("counter=")) counter = Long.parseUnsignedLong(part.substring(8), 16);
                else if (part.startsWith("probability=")) probability = Long.parseUnsignedLong(part.substring(12), 16);
            }
            int count = ir.values().size();
            int[] ranks = new int[count];
            int[] offsets = new int[count];
            int offset = 4;
            for (int boundary = 0; boundary < count; boundary++) {
                ranks[boundary] = ir.values().get(boundary).accessPlan().axisRoles().size();
                offsets[boundary] = offset;
                offset += 2 + 2 * ranks[boundary];
            }
            DataType type = family == CpuRandomIr.Family.INITIAL_STATE
                    ? DataType.INT64 : ir.values().getFirst().dataType();
            return new Encoding(family, type, key, counter, probability, count, ranks, offsets);
        }
    }
}

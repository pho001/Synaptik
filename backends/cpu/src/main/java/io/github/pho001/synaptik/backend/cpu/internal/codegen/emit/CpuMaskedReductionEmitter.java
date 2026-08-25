package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;

/**
 * Emits direct typed masked SUM/MEAN bytecode over complete output-cell ranges.
 *
 * <p>The generated body derives right-aligned mask addresses directly from cold packed layouts.
 * It loads and branches on the canonical mask byte before loading or classifying data, increments
 * one primitive selected-count local only on true, and delegates exact finite/special-value state
 * transitions and represented rounding to {@link CpuExactSumEmitter}. Generated classes do not
 * reference this generation-time type or any Synaptik runtime helper.</p>
 */
public final class CpuMaskedReductionEmitter {
    /** Creates a stateless generation-time emitter. */
    public CpuMaskedReductionEmitter() { }

    /**
     * Emits one exact three-boundary masked-reduction entry body.
     *
     * @param code non-null Class-File method builder to mutate
     * @param specialization non-null typed data/mask/output carrier and scratch specialization
     * @param ir non-null instruction-free masked-reduction identity
     * @throws NullPointerException if {@code code}, {@code specialization}, or {@code ir} is
     *     {@code null}
     * @throws IllegalArgumentException if boundary, scratch, type, or family facts disagree
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        if (specialization.carrierPattern().size() != 3
                || specialization.boundaryDataTypes().size() != 3
                || specialization.boundaryDataTypes().get(1) != DataType.BOOL
                || specialization.boundaryDataTypes().get(0)
                    != specialization.boundaryDataTypes().get(2)
                || !specialization.scratchParameter()) {
            throw new IllegalArgumentException("masked reduction requires typed data/mask/output and scratch");
        }
        String identity = ir.familyIdentity();
        if (!identity.startsWith("masked-reduction:")) throw new IllegalArgumentException(
                "masked-reduction family identity required");
        DataType type = specialization.boundaryDataTypes().getFirst();
        boolean mean = identity.startsWith("masked-reduction:MEAN:");
        int axis = integerAfter(identity, ":axis=");
        int dataRank = ir.values().get(0).accessPlan().iterationRank();
        int maskRank = ir.values().get(1).accessPlan().iterationRank();
        int outputRank = ir.values().get(2).accessPlan().iterationRank();
        int limbs = integerAfter(identity, ":limbs=");
        int dataLayout = 11;
        int maskLayout = dataLayout + 2 + 2 * dataRank;
        int outputLayout = maskLayout + 2 + 2 * maskRank;
        int scratchSlot = 3, geometrySlot = 4, startSlot = 5, endSlot = 7;
        int cell = code.allocateLocal(TypeKind.LONG);
        int remaining = code.allocateLocal(TypeKind.LONG);
        int coordinate = code.allocateLocal(TypeKind.LONG);
        int dataBase = code.allocateLocal(TypeKind.LONG);
        int maskBase = code.allocateLocal(TypeKind.LONG);
        int outputAddress = code.allocateLocal(TypeKind.LONG);
        int dataAddress = code.allocateLocal(TypeKind.LONG);
        int maskAddress = code.allocateLocal(TypeKind.LONG);
        int domainIndex = code.allocateLocal(TypeKind.LONG);
        int selectedCount = code.allocateLocal(TypeKind.LONG);
        int selected = code.allocateLocal(TypeKind.INT);
        int value = code.allocateLocal(localKind(type));
        int result = code.allocateLocal(localKind(type));
        var carriers = new CpuCarrierEmitter(code);
        var exact = new CpuExactSumEmitter(code, type, mean, scratchSlot, geometrySlot, limbs);
        code.lload(startSlot).lstore(cell);
        var cells = code.newLabel(); var done = code.newLabel();
        code.labelBinding(cells).lload(cell).lload(endSlot).lcmp().branch(Opcode.IFGE, done);
        geometry(code, geometrySlot, dataLayout + 1).lstore(dataBase);
        geometry(code, geometrySlot, maskLayout + 1).lstore(maskBase);
        geometry(code, geometrySlot, outputLayout + 1).lstore(outputAddress);
        code.lload(cell).lstore(remaining);
        int omitted = dataRank - maskRank;
        for (int outputAxis = outputRank - 1; outputAxis >= 0; outputAxis--) {
            if (outputAxis == 0) {
                code.lload(remaining).lstore(coordinate);
            } else {
                code.lload(remaining);
                geometry(code, geometrySlot, outputLayout + 2 + outputAxis);
                code.lrem().lstore(coordinate);
                code.lload(remaining);
                geometry(code, geometrySlot, outputLayout + 2 + outputAxis);
                code.ldiv().lstore(remaining);
            }
            int dataAxis = outputAxis < axis ? outputAxis : outputAxis + 1;
            code.lload(dataBase).lload(coordinate);
            geometry(code, geometrySlot, dataLayout + 2 + dataRank + dataAxis);
            code.lmul().ladd().lstore(dataBase);
            code.lload(outputAddress).lload(coordinate);
            geometry(code, geometrySlot, outputLayout + 2 + outputRank + outputAxis);
            code.lmul().ladd().lstore(outputAddress);
            if (dataAxis >= omitted) {
                int maskAxis = dataAxis - omitted;
                var singleton = code.newLabel();
                geometry(code, geometrySlot, maskLayout + 2 + maskAxis)
                        .loadConstant(1L).lcmp().branch(Opcode.IFEQ, singleton);
                code.lload(maskBase).lload(coordinate);
                geometry(code, geometrySlot, maskLayout + 2 + maskRank + maskAxis);
                code.lmul().ladd().lstore(maskBase).labelBinding(singleton);
            }
        }
        code.lload(dataBase).lstore(dataAddress);
        code.lload(maskBase).lstore(maskAddress);
        code.loadConstant(0L).lstore(domainIndex);
        code.loadConstant(0L).lstore(selectedCount);
        exact.emitReset();
        var domain = code.newLabel(); var finish = code.newLabel();
        code.labelBinding(domain).lload(domainIndex);
        geometry(code, geometrySlot, 7).lcmp().branch(Opcode.IFGE, finish);
        carriers.load(DataType.BOOL, specialization.carrierPattern().get(1), 1,
                maskAddress, false);
        code.istore(selected);
        var skip = code.newLabel();
        code.iload(selected).branch(Opcode.IFEQ, skip);
        code.lload(selectedCount).loadConstant(1L).ladd().lstore(selectedCount);
        carriers.load(type, specialization.carrierPattern().getFirst(), 0, dataAddress, false);
        store(code, type, value);
        exact.emitFactor(value);
        code.labelBinding(skip);
        code.lload(dataAddress);
        geometry(code, geometrySlot, dataLayout + 2 + dataRank + axis);
        code.ladd().lstore(dataAddress);
        int alignedMaskAxis = axis - omitted;
        if (alignedMaskAxis >= 0) {
            var singleton = code.newLabel();
            geometry(code, geometrySlot, maskLayout + 2 + alignedMaskAxis)
                    .loadConstant(1L).lcmp().branch(Opcode.IFEQ, singleton);
            code.lload(maskAddress);
            geometry(code, geometrySlot, maskLayout + 2 + maskRank + alignedMaskAxis);
            code.ladd().lstore(maskAddress).labelBinding(singleton);
        }
        code.lload(domainIndex).loadConstant(1L).ladd().lstore(domainIndex)
                .branch(Opcode.GOTO, domain).labelBinding(finish);
        exact.emitFinish(result, selectedCount);
        carriers.store(type, specialization.carrierPattern().get(2), 2,
                outputAddress, result, false);
        code.lload(cell).loadConstant(1L).ladd().lstore(cell)
                .branch(Opcode.GOTO, cells).labelBinding(done);
    }

    private static CodeBuilder geometry(CodeBuilder code, int slot, int index) {
        return code.aload(slot).loadConstant(index).laload();
    }

    private static int integerAfter(String value, String marker) {
        int start = value.indexOf(marker) + marker.length();
        int end = value.indexOf(':', start);
        return Integer.parseInt(value.substring(start, end));
    }

    private static TypeKind localKind(DataType type) {
        return type == DataType.FLOAT64 ? TypeKind.DOUBLE
                : type == DataType.FLOAT32 ? TypeKind.FLOAT : TypeKind.INT;
    }

    private static void store(CodeBuilder code, DataType type, int local) {
        if (type == DataType.FLOAT64) code.dstore(local);
        else if (type == DataType.FLOAT32) code.fstore(local);
        else code.istore(local);
    }
}

package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr;
import java.lang.classfile.ClassBuilder;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;
import java.lang.reflect.AccessFlag;
import java.util.ArrayList;
import java.util.List;

/**
 * Generation-only orchestration of ordinary physical units in a Scan time loop.
 * Numerical work belongs exclusively to the ordinary same-class helpers. Invocation-owned
 * packs, segment wrappers and scratch slices are bound before entry; generated time steps
 * only reset numeric cursor cells, update presence and make fixed {@code invokestatic} calls.
 * Low helpers retain their declared byte/short boundaries; this orchestration never widens a
 * stored carry, history or public value into a replacement FLOAT32 representation.
 */
final class CpuGeneralScanBodyEmitter {
    private static final ClassDesc SEGMENT = ClassDesc.of("java.lang.foreign.MemorySegment");
    private static final MethodTypeDesc ENTRY = MethodTypeDesc.of(ConstantDescs.CD_void,
            SEGMENT.arrayType(), ConstantDescs.CD_long.arrayType().arrayType(), SEGMENT, SEGMENT,
            ConstantDescs.CD_long, ConstantDescs.CD_long);

    /** Creates generation-only orchestration without invocation storage or executable state. */
    CpuGeneralScanBodyEmitter() {}

    /**
     * Uses ordinary declaration owners for actual emission and inventory-only visits.
     * @param builder destination class builder, or null to collect the expected inventory only
     * @param owner generated class declaring the entry and every namespaced ordinary helper
     * @param prefix exact method-name namespace shared with temporal helper calls
     * @param specialization complete ordered ordinary helper specializations and reset recipes
     * @param ir authenticated composite Scan with matching physical unit and temporal inventories
     * @return immutable ordered helper declarations followed by the composite entry declaration
     * @throws IllegalArgumentException if helper inventory or ordinary specialization facts disagree
     */
    List<CpuClassFileKernelGenerator.GeneratedMethod> emitMethods(ClassBuilder builder,
            ClassDesc owner, String prefix, CpuKernelSpecialization specialization, CpuGeneralScanIr ir) {
        var composition = ir.composition().orElseThrow();
        if (composition.units().size() != specialization.compositionUnits().size())
            throw new IllegalArgumentException("Scan helper specialization inventory disagrees");
        var inventory = new ArrayList<CpuClassFileKernelGenerator.GeneratedMethod>();
        for (int u = 0; u < composition.units().size(); u++) {
            var unit = composition.units().get(u);
            var nested = specialization.compositionUnits().get(u).specialization();
            String name = prefix + "scanUnit" + u + "$";
            inventory.addAll(builder == null
                    ? CpuClassFileKernelGenerator.methodInventory(owner, name, nested, unit.kernel(), true)
                    : CpuClassFileKernelGenerator.emitMethods(builder, owner, name, nested, unit.kernel(), unit.control()));
        }
        String entry = prefix + "invoke";
        int flags = AccessFlag.STATIC.mask() | AccessFlag.PUBLIC.mask();
        if (builder != null) builder.withMethodBody(entry, ENTRY, flags, code -> {
            new CpuGeneralScanEmitter().emitComposite(code, owner, prefix, specialization, ir);
            code.return_();
        });
        inventory.add(new CpuClassFileKernelGenerator.GeneratedMethod(entry, ENTRY, flags));
        return List.copyOf(inventory);
    }

    /**
     * Emits fixed helper orchestration inside the Scan owner's already active time step.
     * Complete unit validation precedes merge copies, cursor resets and numerical helper writes.
     *
     * @param scan generation-only temporal bindings owned by the enclosing Scan emitter
     */
    void emitStep(CpuGeneralScanEmitter.Composite scan) {
        var code = scan.code;
        for (int index = 0; index < scan.composition.units().size(); index++) {
            var unit = scan.composition.units().get(index);
            int scope = unit.accumulator().map(a -> a.contributionScope()).orElse(unit.scope());
            scan.setCurrentControl(unit.merge().map(m -> m.leftScope()).orElse(scope),
                    unit.merge().map(m -> m.rightScope()).orElse(-1));
            scan.validateUnit(index);
            if (unit.merge().isPresent()) {
                var merge = unit.merge().orElseThrow();
                var instruction = unit.kernel().instructions().getFirst();
                int left = scan.localBoundary(unit, instruction.inputs().get(0));
                int right = scan.localBoundary(unit, instruction.inputs().get(1));
                int output = scan.localBoundary(unit, instruction.output());
                scan.copyValue(left, output, merge.leftScope(), merge.rightScope());
                scan.copyValue(right, output, merge.rightScope(), merge.leftScope());
            }
            var emission = scan.specialization.compositionUnits().get(index);
            for (int r = 0; r < emission.resets().size(); r++) {
                var reset = emission.resets().get(r);
                code.aload(1).loadConstant(index).aaload().loadConstant(reset.cell());
                code.aload(1).loadConstant(index).aaload().dup().arraylength()
                        .loadConstant(emission.resets().size()+1-r).isub().laload().lastore();
            }
            var nested = emission.specialization();
            if (nested.movementOperandTable()) code.aload(2);
            else for (int b = 0; b < nested.boundaryDataTypes().size(); b++) code.aload(2);
            if (nested.scratchParameter()) scan.boundary(scan.ir.boundaryAccesses().size() + index);
            code.aload(1).loadConstant(index).aaload().loadConstant(0L);
            // Ordinary numeric prefixes and their reset/end trailers remain owner-defined.
            code.aload(1).loadConstant(index).aaload().dup().arraylength().loadConstant(1).isub().laload();
            code.aload(3);
            var type = MethodTypeDesc.ofDescriptor(nested.entryType().descriptorString());
            if (nested.movementOperandTable()) type = type.changeParameterType(0, SEGMENT);
            type = type.insertParameterTypes(type.parameterCount(), SEGMENT);
            code.invokestatic(scan.owner, scan.prefix + "scanUnit" + index + "$invoke", type);
        }
    }

}

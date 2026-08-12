package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.ClassFile;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.AccessFlag;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Stateless Java 26 Class-File generator for one typed portable CPU unit.
 *
 * <p>It realizes the already-selected scalar body or eligible preferred-species
 * FLOAT32/FLOAT64, signed-integral, canonical-BOOL, or narrowly virtual floating-mask vector
 * body, and retains the scalar body for tails. It does not choose capability, numerical
 * semantics, access structure, strategy, or fallback.</p>
 * Instruction-free affine, movement, indexing, scatter, and fold forms delegate to their focused
 * emitters after the same structural specialization checks.
 */
public final class CpuClassFileKernelGenerator {
    /** Creates a stateless generator with no retained route or specialization state. */
    public CpuClassFileKernelGenerator() { }

    /**
     * Emits deterministic verified bytes for one exact structural specialization.
     *
     * @param specialization non-null typed carrier, strategy, and compatibility facts
     * @param kernelIr non-null canonical typed pointwise IR matching the specialization
     * @return a new deterministic verified class-byte array
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if specialization and IR facts disagree
     */
    public byte[] generateClassBytes(CpuKernelSpecialization specialization, CpuKernelIr kernelIr) {
        validate(specialization, kernelIr);
        ClassDesc owner = ClassDesc.of(CpuGeneratorSchema.generatedBinaryName(specialization));
        MethodTypeDesc type = MethodTypeDesc.ofDescriptor(specialization.entryType().descriptorString());
        byte[] bytes = ClassFile.of().build(owner, classBuilder -> classBuilder
                .withVersion(ClassFile.JAVA_26_VERSION, 0).withFlags(AccessFlag.FINAL)
                .withMethod(CpuGeneratorSchema.ENTRY_NAME, type, AccessFlag.STATIC.mask(), method ->
                        method.withCode(code -> {
                            if (kernelIr.instructions().isEmpty()) {
                                if (kernelIr.familyIdentity().startsWith("fold:")) {
                                    new CpuFoldEmitter().emit(code, specialization);
                                } else if (kernelIr.familyIdentity().startsWith("scatter:")) {
                                    new CpuScatterEmitter().emit(code, specialization);
                                } else if (kernelIr.familyIdentity().startsWith("indexing:")) {
                                    new CpuIndexingEmitter().emit(code, specialization);
                                } else if (kernelIr.familyIdentity().startsWith("movement:")) {
                                    new CpuDataMovementEmitter().emit(code, specialization, kernelIr);
                                } else {
                                    new CpuAffineCopyEmitter().emit(code, specialization, kernelIr);
                                }
                                code.return_();
                                return;
                            }
                            var carriers = new CpuCarrierEmitter(code);
                            var scalar = new CpuScalarEmitter(code);
                            var loops = new CpuLoopEmitter(code);
                            List<CpuKernelIr.Value> boundaries = kernelIr.values().stream()
                                    .filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL)
                                    .toList();
                            List<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan> plans =
                                    boundaries.stream().map(CpuKernelIr.Value::accessPlan).toList();
                            int[] scalarLocals = allocateScalarLocals(code, kernelIr);
                            java.util.function.Consumer<CpuLoopEmitter.State> scalarBody = state -> {
                                for (int boundary = 0; boundary < boundaries.size(); boundary++) {
                                    CpuKernelIr.Value value = boundaries.get(boundary);
                                    if (value.kind() != CpuKernelIr.Value.Kind.INPUT
                                            || !requiresInputLoad(kernelIr, value.ordinal())) continue;
                                    carriers.load(value.dataType(), specialization.carrierPattern().get(boundary),
                                            boundary, state.addresses()[boundary]);
                                    store(code, value.dataType(), scalarLocals[value.ordinal()]);
                                }
                                kernelIr.instructions().forEach(instruction ->
                                        scalar.emit(kernelIr, instruction, scalarLocals));
                                for (CpuKernelIr.Store store : kernelIr.stores()) {
                                    CpuKernelIr.Value value = kernelIr.values().get(store.value());
                                    int boundary = boundaryIndex(boundaries, value.ordinal());
                                    carriers.store(value.dataType(), specialization.carrierPattern().get(boundary),
                                            boundary, state.addresses()[boundary], scalarLocals[value.ordinal()]);
                                }
                            };
                            if (specialization.executionStrategy().compute()
                                    == io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR) {
                                DataType vectorType = vectorDataType(kernelIr);
                                int[] vectorLocals = allocateVectorLocals(code, kernelIr);
                                var vectorInstructions = new CpuVectorInstructionEmitter(
                                        code, kernelIr, vectorType);
                                loops.emitVector(plans, specialization.vectorSpeciesBitSize()
                                                / vectorType.bitWidth(),
                                        state -> emitVectorBody(code, carriers, vectorInstructions,
                                                specialization, kernelIr, boundaries, state,
                                                vectorLocals, vectorType), scalarBody);
                            } else loops.emit(plans, scalarBody);
                            code.return_();
                        })));
        verify(specialization, bytes);
        return bytes;
    }

    /**
     * Verifies, defines, and resolves one exact generated entry.
     *
     * @param specialization non-null specialization expected by the bytes
     * @param classBytes non-null complete generated class bytes; copied into the returned artifact
     * @return a new strongly owning generated-kernel artifact with the exact direct handle
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if verification, hidden-class definition, or exact handle
     *     resolution fails
     */
    public CpuGeneratedKernel defineClassBytes(CpuKernelSpecialization specialization,
            byte[] classBytes) {
        Objects.requireNonNull(specialization, "specialization");
        Objects.requireNonNull(classBytes, "classBytes");
        verify(specialization, classBytes);
        try {
            MethodHandles.Lookup lookup = MethodHandles.lookup().defineHiddenClass(
                    classBytes, false, MethodHandles.Lookup.ClassOption.NESTMATE);
            var handle = lookup.findStatic(lookup.lookupClass(), CpuGeneratorSchema.ENTRY_NAME,
                    specialization.entryType());
            return new CpuGeneratedKernel(specialization, lookup, handle, classBytes);
        } catch (ReflectiveOperationException | IllegalArgumentException failure) {
            throw new IllegalArgumentException("generated kernel definition failed", failure);
        }
    }

    private static int[] allocateScalarLocals(java.lang.classfile.CodeBuilder code, CpuKernelIr ir) {
        int[] result = new int[ir.values().size()];
        for (CpuKernelIr.Value value : ir.values()) result[value.ordinal()] = code.allocateLocal(
                switch (value.dataType()) {
                    case FLOAT64 -> TypeKind.DOUBLE; case FLOAT32 -> TypeKind.FLOAT;
                    case INT32, BOOL -> TypeKind.INT; case INT64 -> TypeKind.LONG;
                    default -> throw new IllegalArgumentException("unsupported generated type");
                });
        return result;
    }

    private static int[] allocateVectorLocals(java.lang.classfile.CodeBuilder code, CpuKernelIr ir) {
        int[] result = new int[ir.values().size()];
        for (CpuKernelIr.Value value : ir.values()) result[value.ordinal()] =
                code.allocateLocal(TypeKind.REFERENCE);
        return result;
    }

    private static void emitVectorBody(java.lang.classfile.CodeBuilder code,
            CpuCarrierEmitter carriers, CpuVectorInstructionEmitter instructions,
            CpuKernelSpecialization specialization, CpuKernelIr ir,
            List<CpuKernelIr.Value> boundaries, CpuLoopEmitter.State state, int[] locals,
            DataType vectorType) {
        for (int boundary = 0; boundary < boundaries.size(); boundary++) {
            CpuKernelIr.Value value = boundaries.get(boundary);
            if (value.kind() != CpuKernelIr.Value.Kind.INPUT
                    || !requiresInputLoad(ir, value.ordinal())) continue;
            if (value.dataType() == DataType.BOOL && vectorType != DataType.BOOL) {
                carriers.scalarBoolMaskLoad(vectorType,
                        specialization.carrierPattern().get(boundary), boundary,
                        state.addresses()[boundary]);
            } else carriers.vectorLoad(vectorType,
                    specialization.carrierPattern().get(boundary), boundary,
                    state.addresses()[boundary], value.accessPlan().regime()
                            == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.SCALAR_ALL_ZERO);
            code.astore(locals[value.ordinal()]);
        }
        ir.instructions().forEach(instruction -> instructions.emit(instruction, locals));
        for (CpuKernelIr.Store store : ir.stores()) {
            CpuKernelIr.Value value = ir.values().get(store.value());
            int boundary = boundaryIndex(boundaries, value.ordinal());
            carriers.vectorStore(vectorType, specialization.carrierPattern().get(boundary), boundary,
                    state.addresses()[boundary], locals[value.ordinal()]);
        }
    }

    private static int boundaryIndex(List<CpuKernelIr.Value> boundaries, int ordinal) {
        for (int i = 0; i < boundaries.size(); i++) if (boundaries.get(i).ordinal() == ordinal) return i;
        throw new IllegalArgumentException("stored value is not a materialized boundary");
    }

    private static boolean requiresInputLoad(CpuKernelIr ir, int ordinal) {
        return ir.instructions().stream().anyMatch(instruction -> {
            for (int input = 0; input < instruction.inputs().size(); input++) {
                if (instruction.inputs().get(input) != ordinal) continue;
                if (input == 0 && instruction.opcode() == CpuPointwiseOpcode.SCALAR_POW
                        && instruction.powerRealization()
                            == CpuKernelIr.PowerRealization.POSITIVE_ONE) continue;
                return true;
            }
            return false;
        });
    }

    private static void store(java.lang.classfile.CodeBuilder code, DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.dstore(local); case FLOAT32 -> code.fstore(local);
            case INT32, BOOL -> code.istore(local); case INT64 -> code.lstore(local);
            default -> throw new IllegalArgumentException("unsupported generated type");
        }
    }

    private static void validate(CpuKernelSpecialization specialization, CpuKernelIr kernelIr) {
        Objects.requireNonNull(specialization, "specialization");
        Objects.requireNonNull(kernelIr, "kernelIr");
        if (!specialization.loweringFingerprint().hex().equals(kernelIr.structuralKey())) {
            throw new IllegalArgumentException("kernel IR does not match specialization");
        }
        List<DataType> boundaryTypes = kernelIr.values().stream()
                .filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL)
                .map(CpuKernelIr.Value::dataType).toList();
        if (!boundaryTypes.equals(specialization.boundaryDataTypes())
                || kernelIr.instructions().size() > 8
                || kernelIr.stores().size() != 1) {
            throw new IllegalArgumentException("unsupported canonical pointwise IR");
        }
        if (kernelIr.instructions().isEmpty()) {
            boolean indexing = kernelIr.familyIdentity().startsWith("indexing:");
            boolean scatter = kernelIr.familyIdentity().startsWith("scatter:");
            boolean fold = kernelIr.familyIdentity().startsWith("fold:");
            boolean movement = kernelIr.familyIdentity().startsWith("movement:");
            boolean affine = kernelIr.familyIdentity().startsWith("affine:");
            if ((!movement && !affine && !indexing && !scatter && !fold)
                    || affine && kernelIr.values().size() != 2
                    || movement && (kernelIr.values().size() < 2 || kernelIr.values().size() > 17)
                    || kernelIr.values().subList(0, kernelIr.values().size() - 1).stream()
                        .anyMatch(value -> value.kind() != CpuKernelIr.Value.Kind.INPUT)
                    || kernelIr.values().getLast().kind() != CpuKernelIr.Value.Kind.OUTPUT
                    || !indexing && !scatter && !fold && kernelIr.values().stream().map(CpuKernelIr.Value::dataType).distinct().count() != 1
                    || specialization.executionStrategy().compute()
                        != io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute.SCALAR
                    || !specialization.scalarPowerRealizations().isEmpty()) {
                throw new IllegalArgumentException("unsupported canonical affine copy IR");
            }
            return;
        }
        if (!kernelIr.familyIdentity().equals("pointwise")
                || kernelIr.values().stream().anyMatch(value -> value.dataType()
                        == io.github.pho001.synaptik.model.datatype.DataType.BFLOAT16)
                || specialization.carrierPattern().contains(
                        CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY)) {
            throw new IllegalArgumentException("unsupported canonical pointwise IR");
        }
        List<CpuKernelIr.PowerRealization> realizations = kernelIr.instructions().stream()
                .filter(instruction -> instruction.opcode() == CpuPointwiseOpcode.SCALAR_POW)
                .map(CpuKernelIr.Instruction::powerRealization).toList();
        if (!realizations.equals(specialization.scalarPowerRealizations())) {
            throw new IllegalArgumentException("specialization power realizations do not match IR");
        }
        if (specialization.executionStrategy().compute()
                == io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR
                && vectorDataTypeOrNull(kernelIr) == null) {
            throw new IllegalArgumentException(
                    "vector specialization requires one supported typed value or virtual-mask topology");
        }
    }

    private static DataType vectorDataType(CpuKernelIr kernelIr) {
        DataType result = vectorDataTypeOrNull(kernelIr);
        if (result == null) throw new IllegalArgumentException(
                "vector specialization requires one supported typed value or virtual-mask topology");
        return result;
    }

    private static DataType vectorDataTypeOrNull(CpuKernelIr kernelIr) {
        var numeric = kernelIr.values().stream().map(CpuKernelIr.Value::dataType)
                .filter(type -> type != DataType.BOOL).distinct().toList();
        DataType result;
        if (numeric.isEmpty()) result = DataType.BOOL;
        else if (numeric.size() == 1) result = numeric.getFirst();
        else return null;
        if (result != DataType.FLOAT32 && result != DataType.FLOAT64
                && result != DataType.INT32 && result != DataType.INT64
                && result != DataType.BOOL) return null;
        return vectorTopologyEligible(kernelIr, result) ? result : null;
    }

    private static boolean vectorTopologyEligible(CpuKernelIr ir, DataType laneType) {
        boolean mixedMasks = (laneType == DataType.FLOAT32 || laneType == DataType.FLOAT64)
                && ir.values().stream().anyMatch(value -> value.dataType() == DataType.BOOL);
        for (CpuKernelIr.Value value : ir.values()) {
            if (value.dataType() == DataType.BOOL && mixedMasks
                    && value.kind() != CpuKernelIr.Value.Kind.VIRTUAL
                    && !(value.kind() == CpuKernelIr.Value.Kind.INPUT
                        && value.accessPlan().regime()
                            == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.SCALAR_ALL_ZERO)) {
                return false;
            }
        }
        for (CpuKernelIr.Instruction instruction : ir.instructions()) {
            if (!instruction.opcode().vectorEligible()
                    || instruction.opcode() == CpuPointwiseOpcode.SCALAR_POW
                        && instruction.powerRealization() == CpuKernelIr.PowerRealization.DIRECT) {
                return false;
            }
            switch (instruction.opcode().vectorForm()) {
                case VALUE -> {
                    if (!valueOpcodeEligible(instruction.opcode(), laneType)) return false;
                }
                case MASK_PRODUCER -> {
                    if (!mixedMasks || ir.values().get(instruction.output()).kind()
                            != CpuKernelIr.Value.Kind.VIRTUAL) return false;
                }
                case VALUE_OR_MASK -> {
                    boolean byteValues = laneType == DataType.BOOL;
                    boolean virtualMasks = mixedMasks
                            && instruction.inputs().stream().allMatch(input ->
                                ir.values().get(input).kind() == CpuKernelIr.Value.Kind.VIRTUAL)
                            && ir.values().get(instruction.output()).kind()
                                == CpuKernelIr.Value.Kind.VIRTUAL;
                    if (!byteValues && !virtualMasks) return false;
                }
                case MASK_CONSUMER -> {
                    if (!mixedMasks) return false;
                    CpuKernelIr.Value condition = ir.values().get(instruction.inputs().getFirst());
                    if (condition.kind() != CpuKernelIr.Value.Kind.VIRTUAL
                            && !(condition.kind() == CpuKernelIr.Value.Kind.INPUT
                                && condition.accessPlan().regime()
                                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.SCALAR_ALL_ZERO)) {
                        return false;
                    }
                }
                case NONE -> { return false; }
            }
        }
        return true;
    }

    private static boolean valueOpcodeEligible(CpuPointwiseOpcode opcode, DataType laneType) {
        if (laneType == DataType.BOOL) return opcode == CpuPointwiseOpcode.CAST;
        if (laneType == DataType.INT32 || laneType == DataType.INT64) return switch (opcode) {
            case ADD, SUB, MUL, MIN, MAX, SCALAR_ADD, SCALAR_SUB, SCALAR_MUL,
                    SCALAR_MIN, SCALAR_MAX, CAST -> true;
            default -> false;
        };
        return laneType == DataType.FLOAT32 || laneType == DataType.FLOAT64;
    }

    private static void verify(CpuKernelSpecialization specialization, byte[] bytes) {
        try {
            var errors = ClassFile.of().verify(bytes);
            if (!errors.isEmpty()) throw new IllegalArgumentException(
                    "generated class verification failed: " + errors);
            var model = ClassFile.of().parse(bytes);
            String expectedName = CpuGeneratorSchema.generatedBinaryName(specialization).replace('.', '/');
            boolean method = model.methods().size() == 1
                    && model.methods().getFirst().methodName().stringValue().equals(CpuGeneratorSchema.ENTRY_NAME)
                    && model.methods().getFirst().methodType().stringValue()
                    .equals(specialization.entryType().descriptorString());
            if (model.majorVersion() != ClassFile.JAVA_26_VERSION
                    || !model.thisClass().asInternalName().equals(expectedName)
                    || !model.interfaces().isEmpty() || !model.fields().isEmpty() || !method) {
                throw new IllegalArgumentException("generated class shape does not match specialization");
            }
        } catch (IllegalArgumentException failure) { throw failure; }
        catch (RuntimeException failure) {
            throw new IllegalArgumentException("generated class verification failed", failure);
        }
    }
}

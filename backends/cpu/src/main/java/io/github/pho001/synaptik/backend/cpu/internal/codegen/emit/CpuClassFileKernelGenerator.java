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

/** Stateless Java 26 Class-File generator for one typed pointwise CPU unit. */
public final class CpuClassFileKernelGenerator {
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
                                    if (value.kind() != CpuKernelIr.Value.Kind.INPUT) continue;
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
                                int[] vectorLocals = allocateVectorLocals(code, kernelIr);
                                loops.emitVector(plans, specialization.vectorSpeciesBitSize() / Double.SIZE,
                                        state -> emitVectorBody(code, carriers, specialization, kernelIr,
                                                boundaries, state, vectorLocals), scalarBody);
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
            CpuCarrierEmitter carriers, CpuKernelSpecialization specialization, CpuKernelIr ir,
            List<CpuKernelIr.Value> boundaries, CpuLoopEmitter.State state, int[] locals) {
        ClassDesc vector = ClassDesc.of("jdk.incubator.vector.DoubleVector");
        ClassDesc vectorBase = ClassDesc.of("jdk.incubator.vector.Vector");
        for (int boundary = 0; boundary < boundaries.size(); boundary++) {
            CpuKernelIr.Value value = boundaries.get(boundary);
            if (value.kind() != CpuKernelIr.Value.Kind.INPUT) continue;
            carriers.vectorLoad(specialization.carrierPattern().get(boundary), boundary,
                    state.addresses()[boundary], value.accessPlan().regime()
                            == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.SCALAR_ALL_ZERO);
            code.astore(locals[value.ordinal()]);
        }
        for (CpuKernelIr.Instruction instruction : ir.instructions()) {
            code.aload(locals[instruction.inputs().getFirst()]);
            switch (instruction.opcode()) {
                case ADD, SUB, MUL -> {
                    code.aload(locals[instruction.inputs().get(1)]);
                    String method = instruction.opcode() == CpuPointwiseOpcode.ADD ? "add"
                            : instruction.opcode() == CpuPointwiseOpcode.SUB ? "sub" : "mul";
                    code.invokevirtual(vector, method, MethodTypeDesc.of(vector, vectorBase));
                }
                case SCALAR_ADD, SCALAR_SUB, SCALAR_MUL -> {
                    code.loadConstant(Double.longBitsToDouble(instruction.scalarImmediate().bits()));
                    String method = instruction.opcode() == CpuPointwiseOpcode.SCALAR_ADD ? "add"
                            : instruction.opcode() == CpuPointwiseOpcode.SCALAR_SUB ? "sub" : "mul";
                    code.invokevirtual(vector, method, MethodTypeDesc.of(vector,
                            java.lang.constant.ConstantDescs.CD_double));
                }
                case NEG -> code.invokevirtual(vector, "neg", MethodTypeDesc.of(vector));
                case GELU_EXACT -> code.invokestatic(ClassDesc.of(CpuVectorEmitter.class.getName()),
                        "gelu", MethodTypeDesc.of(vector, vector));
                default -> throw new IllegalArgumentException("unsupported vector opcode");
            }
            code.astore(locals[instruction.output()]);
        }
        for (CpuKernelIr.Store store : ir.stores()) {
            CpuKernelIr.Value value = ir.values().get(store.value());
            int boundary = boundaryIndex(boundaries, value.ordinal());
            carriers.vectorStore(specialization.carrierPattern().get(boundary), boundary,
                    state.addresses()[boundary], locals[value.ordinal()]);
        }
    }

    private static int boundaryIndex(List<CpuKernelIr.Value> boundaries, int ordinal) {
        for (int i = 0; i < boundaries.size(); i++) if (boundaries.get(i).ordinal() == ordinal) return i;
        throw new IllegalArgumentException("stored value is not a materialized boundary");
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
                || kernelIr.instructions().isEmpty() || kernelIr.instructions().size() > 8
                || kernelIr.stores().size() != 1) {
            throw new IllegalArgumentException("unsupported canonical pointwise IR");
        }
        if (specialization.executionStrategy().compute()
                == io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR
                && (kernelIr.values().stream().anyMatch(value -> value.dataType() != DataType.FLOAT64)
                    || kernelIr.instructions().stream().anyMatch(instruction ->
                            !instruction.opcode().vectorEligible()))) {
            throw new IllegalArgumentException("vector specialization requires FLOAT64 numeric-only IR");
        }
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

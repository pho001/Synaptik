package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import java.lang.classfile.ClassFile;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.AccessFlag;
import java.util.Objects;

/**
 * Stateless Java 26 Class-File API generator and verifier for the canonical fused CPU unit.
 */
public final class CpuClassFileKernelGenerator {
    /**
     * Emits deterministic bytes for the route-independent canonical IR.
     * @param specialization non-null exact/default scalar-or-vector structural specialization
     * @param kernelIr non-null matching canonical IR with the exact ordered fused semantics
     * @return a new deterministic verified class-byte array; never {@code null}
     * @throws NullPointerException if either argument is {@code null}
     * @throws IllegalArgumentException if the IR, specialization, generated version, or generated
     *     class shape is incompatible
     */
    public byte[] generateClassBytes(CpuKernelSpecialization specialization, CpuKernelIr kernelIr) {
        validate(specialization, kernelIr);
        ClassDesc owner = ClassDesc.of(CpuGeneratorSchema.generatedBinaryName(specialization));
        MethodTypeDesc type = MethodTypeDesc.ofDescriptor(
                specialization.entryType().descriptorString());
        byte[] bytes = ClassFile.of().build(owner, classBuilder -> classBuilder
                .withVersion(ClassFile.JAVA_26_VERSION, 0)
                .withFlags(AccessFlag.FINAL)
                .withMethod(CpuGeneratorSchema.ENTRY_NAME, type, AccessFlag.STATIC.mask(), method ->
                        method.withCode(code -> {
                            var carriers = new CpuCarrierEmitter(code);
                            var scalar = new CpuScalarEmitter(code);
                            var loops = new CpuLoopEmitter(code);
                            var plans = java.util.List.of(kernelIr.values().get(0).accessPlan(),
                                    kernelIr.values().get(1).accessPlan(),
                                    kernelIr.values().get(2).accessPlan(),
                                    kernelIr.values().get(5).accessPlan());
                            java.util.function.Consumer<CpuLoopEmitter.State> scalarBody = state -> {
                                carriers.load(specialization.carrierPattern().get(0), 0,
                                        state.addresses()[0]);
                                carriers.load(specialization.carrierPattern().get(1), 1,
                                        state.addresses()[1]);
                                code.dadd(); scalar.gelu();
                                carriers.load(specialization.carrierPattern().get(2), 2,
                                        state.addresses()[2]);
                                code.dmul();
                                int result = code.allocateLocal(TypeKind.DOUBLE);
                                code.dstore(result);
                                carriers.store(specialization.carrierPattern().get(3), 3,
                                        state.addresses()[3], result);
                            };
                            if (specialization.executionStrategy().compute()
                                    == io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR) {
                                ClassDesc vector = ClassDesc.of("jdk.incubator.vector.DoubleVector");
                                ClassDesc vectorBase = ClassDesc.of("jdk.incubator.vector.Vector");
                                loops.emitVector(plans,
                                        specialization.vectorSpeciesBitSize() / Double.SIZE,
                                        state -> {
                                            carriers.vectorLoad(specialization.carrierPattern().get(0),
                                                    0, state.addresses()[0], plans.get(0).regime()
                                                            == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.SCALAR_ALL_ZERO);
                                            carriers.vectorLoad(specialization.carrierPattern().get(1),
                                                    1, state.addresses()[1], plans.get(1).regime()
                                                            == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.SCALAR_ALL_ZERO);
                                            code.invokevirtual(vector, "add", MethodTypeDesc.of(
                                                    vector, vectorBase));
                                            code.invokestatic(ClassDesc.of(CpuVectorEmitter.class.getName()),
                                                    "gelu", MethodTypeDesc.of(vector, vector));
                                            carriers.vectorLoad(specialization.carrierPattern().get(2),
                                                    2, state.addresses()[2], plans.get(2).regime()
                                                            == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.SCALAR_ALL_ZERO);
                                            code.invokevirtual(vector, "mul", MethodTypeDesc.of(
                                                    vector, vectorBase));
                                            int result = code.allocateLocal(TypeKind.REFERENCE);
                                            code.astore(result);
                                            carriers.vectorStore(specialization.carrierPattern().get(3),
                                                    3, state.addresses()[3], result);
                                        }, scalarBody);
                                code.return_();
                                return;
                            }
                            loops.emit(plans, scalarBody);
                            code.return_();
                        })));
        verify(specialization, bytes);
        return bytes;
    }

    /**
     * Verifies and defines compatible bytes, then resolves the exact static entry point.
     * @param specialization non-null exact structural specialization
     * @param classBytes non-null class bytes to verify before hidden-class definition; not retained
     *     directly
     * @return a loaded, strongly owned generated artifact with a defensive byte snapshot
     * @throws NullPointerException if either argument is {@code null}
     * @throws IllegalArgumentException if verification, definition, or entry-point resolution fails
     */
    public CpuGeneratedKernel defineClassBytes(
            CpuKernelSpecialization specialization, byte[] classBytes) {
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

    private static void validate(CpuKernelSpecialization specialization, CpuKernelIr kernelIr) {
        Objects.requireNonNull(specialization, "specialization");
        Objects.requireNonNull(kernelIr, "kernelIr");
        if (!specialization.loweringFingerprint().hex().equals(kernelIr.structuralKey())) {
            throw new IllegalArgumentException("kernel IR does not match specialization");
        }
        var expected = java.util.List.of(CpuKernelIr.Instruction.Semantic.ADD,
                CpuKernelIr.Instruction.Semantic.GELU_EXACT,
                CpuKernelIr.Instruction.Semantic.MUL);
        if (!kernelIr.instructions().stream().map(CpuKernelIr.Instruction::semantic).toList()
                .equals(expected)) throw new IllegalArgumentException("unsupported canonical IR");
    }

    private static void verify(CpuKernelSpecialization specialization, byte[] bytes) {
        try {
            var errors = ClassFile.of().verify(bytes);
            if (!errors.isEmpty()) throw new IllegalArgumentException(
                    "generated class verification failed: " + errors);
            var model = ClassFile.of().parse(bytes);
            String expectedName = CpuGeneratorSchema.generatedBinaryName(specialization)
                    .replace('.', '/');
            boolean method = model.methods().size() == 1
                    && model.methods().getFirst().methodName().stringValue()
                            .equals(CpuGeneratorSchema.ENTRY_NAME)
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

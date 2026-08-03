package io.github.pho001.synaptik.backend.cpu.execution;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.classfile.ClassFile;
import java.lang.reflect.Modifier;
import java.lang.reflect.AccessFlag;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuGeneratedKernelShapeTest {
    @Test void generatedClassHasJava26VersionNoFieldsAndOneStaticTypedMethod() {
        var specialization = CpuKernelSpecializationTest.specialization(
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD, List.of(), List.of(), 0, null,
                CpuKernelSpecialization.Tail.NONE, 1);
        CpuFamilyKernelEmitter emitter = new CpuFamilyKernelEmitter() {
            @Override public CpuLoweringFingerprint loweringFingerprint() {
                return specialization.loweringFingerprint();
            }
            @Override public void emitScalar(CpuScalarEmitter scalar, CpuCarrierEmitter carriers,
                    CpuLoopEmitter loops, CpuReductionEmitter reductions) { scalar.code().return_(); }
            @Override public void emitVector(CpuVectorEmitter vector, CpuCarrierEmitter carriers,
                    CpuLoopEmitter loops, CpuReductionEmitter reductions) { vector.code().return_(); }
        };
        CpuGeneratedKernel kernel = new CpuClassFileKernelGenerator().generate(specialization, emitter);
        var model = ClassFile.of().parse(kernel.classBytes());
        assertEquals(ClassFile.JAVA_26_VERSION, model.majorVersion());
        assertTrue(model.fields().isEmpty()); assertEquals(1, model.methods().size());
        assertEquals("invoke", model.methods().getFirst().methodName().stringValue());
        assertEquals(specialization.entryType().descriptorString(),
                model.methods().getFirst().methodType().stringValue());
        assertTrue(model.methods().getFirst().flags().has(AccessFlag.STATIC));
        assertTrue(ClassFile.of().verify(kernel.classBytes()).isEmpty());
    }

    @Test void allNewProductionTypesRemainPackagePrivateAndArtifactUsesIdentityEquality() throws Exception {
        Class<?>[] types = {CpuGeneratorSchema.class, CpuPortableExecutionMode.class,
                CpuLoweringFingerprint.class, CpuKernelSpecialization.class,
                CpuFamilyKernelEmitter.class, CpuScalarEmitter.class, CpuVectorEmitter.class,
                CpuCarrierEmitter.class, CpuLoopEmitter.class, CpuReductionEmitter.class,
                CpuClassFileKernelGenerator.class, CpuGeneratedKernel.class};
        for (Class<?> type : types) assertFalse(Modifier.isPublic(type.getModifiers()), type.getName());
        assertSame(Object.class, CpuGeneratedKernel.class.getMethod("equals", Object.class).getDeclaringClass());
        assertSame(Object.class, CpuGeneratedKernel.class.getMethod("hashCode").getDeclaringClass());
    }
}

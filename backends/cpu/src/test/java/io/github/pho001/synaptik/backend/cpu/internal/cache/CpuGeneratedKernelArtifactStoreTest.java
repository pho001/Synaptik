package io.github.pho001.synaptik.backend.cpu.internal.cache;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.model.shape.Shape;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CpuGeneratedKernelArtifactStoreTest {
    @TempDir Path root;

    @Test void supportsNoRootCurrentSchemaHitsAndCorruptRootFallback() throws Exception {
        var route = CpuPartitionPreparerTest.analyze(Shape.of(8)).plan().units().getFirst().portablePlan();
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var memoryOnly = new CpuGeneratedKernelArtifactStore().loadOrGenerate(
                route.specialization(), route.kernelIr());
        Files.write(root.resolve("legacy-v1.class"), new byte[]{1, 2, 3});
        Path current = root.resolve(route.specialization().structuralKey() + ".class");
        Files.write(current, new byte[]{1, 2, 3});
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var persisted = new CpuGeneratedKernelArtifactStore(Optional.of(root)).loadOrGenerate(
                route.specialization(), route.kernelIr());
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var hit = new CpuGeneratedKernelArtifactStore(Optional.of(root)).loadOrGenerate(
                route.specialization(), route.kernelIr());
        assertAll(
                () -> assertTrue(Files.exists(root.resolve("legacy-v1.class"))),
                () -> assertArrayEquals(memoryOnly.classBytes(), persisted.classBytes()),
                () -> assertArrayEquals(persisted.classBytes(), Files.readAllBytes(current)),
                () -> assertArrayEquals(persisted.classBytes(), hit.classBytes()),
                () -> assertNotSame(persisted.hiddenClass(), hit.hiddenClass()));
    }
}

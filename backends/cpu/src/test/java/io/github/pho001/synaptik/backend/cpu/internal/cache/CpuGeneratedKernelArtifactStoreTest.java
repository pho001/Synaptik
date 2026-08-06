package io.github.pho001.synaptik.backend.cpu.internal.cache;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.model.shape.Shape;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import jdk.incubator.vector.DoubleVector;

class CpuGeneratedKernelArtifactStoreTest {
    @TempDir Path root;

    @Test void supportsNoRootCurrentSchemaHitsAndCorruptRootFallback() throws Exception {
        var route = CpuPartitionPreparerTest.analyze(Shape.of(8)).plan().units().getFirst().portablePlan();
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var memoryOnly = new CpuGeneratedKernelArtifactStore().loadOrGenerate(
                route.specialization(), route.kernelIr());
        Files.write(root.resolve("legacy-v1.class"), new byte[]{1, 2, 3});
        Path current = root.resolve(route.specialization().structuralKey() + ".class");
        Path metadata = root.resolve(route.specialization().structuralKey() + ".meta");
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
                () -> assertArrayEquals(route.specialization().compatibilityBytes(),
                        Files.readAllBytes(metadata)),
                () -> assertArrayEquals(persisted.classBytes(), hit.classBytes()),
                () -> assertNotSame(persisted.hiddenClass(), hit.hiddenClass()));
    }

    @Test void rejectsIncompatibleAndCorruptPersistedVectorSpeciesMetadata() throws Exception {
        int lanes = DoubleVector.SPECIES_PREFERRED.length();
        var descriptor = CpuPartitionPreparerTest.context(Shape.of(lanes * 2));
        var vectorInputs = new CpuPartitionAnalysisInputs(false,
                CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 1, 1, 1));
        var vectorContext = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                descriptor.partition(), descriptor.nodes(), descriptor.values(),
                descriptor.memoryRequirements(), descriptor.constants(), vectorInputs);
        var vectorRoute = new io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer()
                .analyze(vectorContext).plan().units().getFirst().portablePlan();
        var scalarRoute = CpuPartitionPreparerTest.analyze(Shape.of(lanes * 2)).plan().units()
                .getFirst().portablePlan();
        var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));
        store.loadOrGenerate(vectorRoute.specialization(), vectorRoute.kernelIr());
        Path metadata = root.resolve(vectorRoute.specialization().structuralKey() + ".meta");

        Files.write(metadata, scalarRoute.specialization().compatibilityBytes());
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        store.loadOrGenerate(vectorRoute.specialization(), vectorRoute.kernelIr());
        assertArrayEquals(vectorRoute.specialization().compatibilityBytes(),
                Files.readAllBytes(metadata));

        Files.write(metadata, new byte[] {1, 2, 3});
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        store.loadOrGenerate(vectorRoute.specialization(), vectorRoute.kernelIr());
        assertArrayEquals(vectorRoute.specialization().compatibilityBytes(),
                Files.readAllBytes(metadata));
    }
}

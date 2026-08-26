package io.github.pho001.synaptik.backend.cpu.internal.cache;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.model.shape.Shape;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.List;
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
        Path current = root.resolve(route.specialization().structuralKey() + ".artifact");
        Files.write(current, new byte[]{1, 2, 3});
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var persistedResult = new CpuGeneratedKernelArtifactStore(Optional.of(root)).loadOrGenerateObserved(
                route.specialization(), route.kernelIr());
        var persisted = persistedResult.artifact();
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var hitResult = new CpuGeneratedKernelArtifactStore(Optional.of(root)).loadOrGenerateObserved(
                route.specialization(), route.kernelIr());
        var hit = hitResult.artifact();
        assertAll(
                () -> assertEquals(51, CpuGeneratorSchema.CURRENT_VERSION),
                () -> assertTrue(Files.exists(root.resolve("legacy-v1.class"))),
                () -> assertArrayEquals(memoryOnly.classBytes(), persisted.classBytes()),
                () -> assertTrue(Files.size(current) > persisted.classBytes().length),
                () -> assertArrayEquals(persisted.classBytes(), hit.classBytes()),
                () -> assertNotSame(persisted.hiddenClass(), hit.hiddenClass()),
                () -> assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,
                        persistedResult.source()),
                () -> assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT,
                        hitResult.source()));
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
        Path envelope = root.resolve(vectorRoute.specialization().structuralKey() + ".artifact");

        Files.write(envelope, scalarRoute.specialization().compatibilityBytes());
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        store.loadOrGenerate(vectorRoute.specialization(), vectorRoute.kernelIr());
        assertTrue(Files.size(envelope) > vectorRoute.specialization().compatibilityBytes().length);

        Files.write(envelope, new byte[] {1, 2, 3});
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        store.loadOrGenerate(vectorRoute.specialization(), vectorRoute.kernelIr());
        assertTrue(Files.size(envelope) > 3);
    }

    @Test void rejectsEveryBoundedEnvelopeFailureWithoutDefiningInvalidBytes() throws Exception {
        var route = CpuPartitionPreparerTest.analyze(Shape.of(8)).plan().units().getFirst()
                .portablePlan();
        var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var seed = store.loadOrGenerateObserved(route.specialization(), route.kernelIr());
        Path file = root.resolve(route.specialization().structuralKey() + ".artifact");
        byte[] valid = Files.readAllBytes(file);
        byte[] trailing = Arrays.copyOf(valid, valid.length + 1);
        byte[] truncated = Arrays.copyOf(valid, valid.length - 1);
        byte[] wrongChecksum = valid.clone();
        wrongChecksum[wrongChecksum.length - 1] ^= 1;
        byte[] schema43 = valid.clone();
        java.nio.ByteBuffer.wrap(schema43).putInt(4, 43);
        byte[] schema42 = valid.clone();
        java.nio.ByteBuffer.wrap(schema42).putInt(4, 42);
        byte[] malformedClass = envelope(route.specialization().structuralKey(),
                route.specialization().compatibilityBytes(), new byte[] {1, 2, 3, 4});
        byte[] wrongMetadata = envelope(route.specialization().structuralKey(),
                new byte[] {9}, seed.artifact().classBytes());
        byte[] wrongKey = envelope("0".repeat(64),
                route.specialization().compatibilityBytes(), seed.artifact().classBytes());
        for (byte[] invalid : List.of(trailing, truncated, wrongChecksum, schema43, schema42,
                malformedClass, wrongMetadata, wrongKey,
                invalidLengthEnvelope(route.specialization().structuralKey(),
                        CpuGeneratedKernelArtifactStore.MAX_METADATA_BYTES + 1, false),
                invalidLengthEnvelope(route.specialization().structuralKey(),
                        route.specialization().compatibilityBytes().length, true),
                new byte[CpuGeneratedKernelArtifactStore.MAX_ENVELOPE_BYTES + 1])) {
            Files.write(file, invalid);
            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            var recovered = store.loadOrGenerateObserved(route.specialization(), route.kernelIr());
            assertAll(
                    () -> assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,
                            recovered.source()),
                    () -> assertArrayEquals(seed.artifact().classBytes(),
                            recovered.artifact().classBytes()),
                    () -> assertEquals(51, java.nio.ByteBuffer.wrap(Files.readAllBytes(file))
                            .getInt(4)),
                    () -> assertTrue(Files.size(file) <=
                            CpuGeneratedKernelArtifactStore.MAX_ENVELOPE_BYTES));
        }
    }

    private static byte[] envelope(String key, byte[] metadata, byte[] classBytes) throws Exception {
        var bytes = new ByteArrayOutputStream();
        try (var output = new DataOutputStream(bytes)) {
            output.writeInt(0x53435055);
            output.writeInt(CpuGeneratorSchema.CURRENT_VERSION);
            output.writeUTF(key);
            output.writeInt(metadata.length);
            output.write(metadata);
            output.writeInt(classBytes.length);
            output.write(classBytes);
            var digest = MessageDigest.getInstance("SHA-256");
            digest.update(key.getBytes(StandardCharsets.US_ASCII));
            digest.update(metadata);
            output.write(digest.digest(classBytes));
        }
        return bytes.toByteArray();
    }

    private static byte[] invalidLengthEnvelope(String key, int metadataLength,
            boolean oversizedClass) throws Exception {
        var bytes = new ByteArrayOutputStream();
        try (var output = new DataOutputStream(bytes)) {
            output.writeInt(0x53435055);
            output.writeInt(CpuGeneratorSchema.CURRENT_VERSION);
            output.writeUTF(key);
            output.writeInt(metadataLength);
            if (oversizedClass) {
                output.write(new byte[metadataLength]);
                output.writeInt(CpuGeneratedKernelArtifactStore.MAX_CLASS_BYTES + 1);
            }
        }
        return bytes.toByteArray();
    }
}

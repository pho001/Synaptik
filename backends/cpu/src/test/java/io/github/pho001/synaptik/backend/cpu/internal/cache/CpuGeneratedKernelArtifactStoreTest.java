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
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;

class CpuGeneratedKernelArtifactStoreTest {
    @TempDir Path root;

    @Test void aggregateSegmentArtifactRejectsPreSchema66EnvelopeAndRegenerates() throws Exception {
        var base = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest
                .context(io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.SUM,
                        DataType.FLOAT32, Shape.of(2, 3),
                        new io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs(1,
                                false), Shape.of(2));
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                                CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                        CpuPartitionAnalysisInputs.DEFAULT.portableExecution()));
        var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
        var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));
        Path envelope = root.resolve(route.specialization().structuralKey() + ".artifact");

        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var initial = store.loadOrGenerateObserved(route.specialization(), route.kernelIr());
        byte[] stale = Files.readAllBytes(envelope);
        java.nio.ByteBuffer.wrap(stale).putInt(4, 65);
        Files.write(envelope, stale);
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var regenerated = store.loadOrGenerateObserved(route.specialization(), route.kernelIr());

        assertAll(
                () -> assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,
                        initial.source()),
                () -> assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,
                        regenerated.source()),
                () -> assertArrayEquals(initial.artifact().classBytes(),
                        regenerated.artifact().classBytes()),
                () -> assertEquals(66, java.nio.ByteBuffer.wrap(Files.readAllBytes(envelope))
                        .getInt(4)));
    }

    /**
     * Proves current-only envelope invalidation against admitted composition routes, rather than
     * a hand-written compatibility payload.
     */
    @Test void concatAndStackRejectSameKeySchema64EnvelopesThenPersistSchema66Hits() throws Exception {
        for (TensorCompositionKind kind : List.of(TensorCompositionKind.CONCAT,
                TensorCompositionKind.STACK)) {
            var route = compositionRoute(kind);
            var specialization = route.specialization();
            var kernelIr = route.kernelIr();
            // Capture immutable facts before persistence.  Post-round-trip assertions must not
            // compare the route object with itself.
            String structuralKey = specialization.structuralKey();
            int classIdentitySchema = specialization.classIdentitySchema();
            String binaryName = CpuGeneratorSchema.generatedBinaryName(specialization);
            String irKey = kernelIr.structuralKey();
            String descriptor = specialization.entryType().descriptorString();
            String compatibilityDigest = java.util.HexFormat.of().formatHex(
                    MessageDigest.getInstance("SHA-256").digest(specialization.compatibilityBytes()));
            var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));
            Path file = root.resolve(structuralKey + ".artifact");

            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            var initial = store.loadOrGenerateObserved(specialization, kernelIr);
            assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,
                    initial.source(), kind + " initial current realization");
            byte[] current = Files.readAllBytes(file);
            byte[] stale = current.clone();
            java.nio.ByteBuffer.wrap(stale).putInt(4, 64);
            Files.write(file, stale);

            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            var regenerated = store.loadOrGenerateObserved(specialization, kernelIr);
            assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,
                    regenerated.source(), kind + " stale same-key envelope is rejected");
            assertEquals(66, java.nio.ByteBuffer.wrap(Files.readAllBytes(file)).getInt(4),
                    kind + " regenerated envelope schema");

            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            var reconstructed = compositionRoute(kind);
            var hit = store.loadOrGenerateObserved(reconstructed.specialization(), reconstructed.kernelIr());
            assertAll(
                    () -> assertEquals(structuralKey, reconstructed.specialization().structuralKey(), kind + " structural key retained"),
                    () -> assertEquals(classIdentitySchema, reconstructed.specialization().classIdentitySchema(), kind + " class identity schema retained"),
                    () -> assertEquals(binaryName, CpuGeneratorSchema.generatedBinaryName(reconstructed.specialization()), kind + " generated binary name retained"),
                    () -> assertEquals(irKey, reconstructed.kernelIr().structuralKey(), kind + " structural IR key retained"),
                    () -> assertEquals(descriptor, reconstructed.specialization().entryType().descriptorString(), kind + " generated descriptor retained"),
                    () -> assertEquals(compatibilityDigest, java.util.HexFormat.of().formatHex(
                            MessageDigest.getInstance("SHA-256").digest(reconstructed.specialization().compatibilityBytes())),
                            kind + " compatibility facts retained"),
                    () -> assertArrayEquals(initial.artifact().classBytes(), regenerated.artifact().classBytes(),
                            kind + " deterministic regenerated class bytes"),
                    () -> assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT,
                            hit.source(), kind + " current envelope persisted hit"),
                    () -> assertArrayEquals(regenerated.artifact().classBytes(), hit.artifact().classBytes(),
                            kind + " persisted class bytes"));
        }
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan compositionRoute(
            TensorCompositionKind kind) {
        var first = CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2));
        var second = CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2));
        var context = CpuNonAffineMovementLoweringTest.context(
                new Operation(kind, new CompositionAxisAttrs(kind == TensorCompositionKind.CONCAT ? 0 : 1)),
                kind == TensorCompositionKind.CONCAT ? List.of(0, 1, 0) : List.of(0, 1),
                List.of(first, second),
                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32,
                        kind == TensorCompositionKind.CONCAT ? Shape.of(6) : Shape.of(2, 2)));
        return new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
    }

    @Test void publishesReloadsSchema66ConvAndRejectsStaleSchema64Envelope() throws Exception {
        var base = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv2dLoweringTest
                .context(List.of(io.github.pho001.synaptik.model.datatype.DataType.FLOAT32,
                                io.github.pho001.synaptik.model.datatype.DataType.FLOAT32),
                        Shape.of(1, 1, 5, 67), Shape.of(2, 1, 3, 3), Shape.of(1, 2, 3, 65),
                        io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs.defaults(),
                        null);
        var carriers = List.of(CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY,
                CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY);
        var vectorInputs = new CpuPartitionAnalysisInputs(false, carriers,
                new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 1, 1, 1));
        var scalarInputs = new CpuPartitionAnalysisInputs(false, carriers,
                new PortableExecutionConfig(ComputePreference.SCALAR, 1, 1, 1));
        var vectorContext = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), vectorInputs);
        var scalarContext = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                base.partition(), base.nodes(), base.values(), base.memoryRequirements(),
                base.constants(), scalarInputs);
        var preparer = new io.github.pho001.synaptik.backend.cpu.internal.prepare
                .CpuPartitionPreparer();
        var vectorRoute = preparer.analyze(vectorContext).plan().units().getFirst().portablePlan();
        var scalarRoute = preparer.analyze(scalarContext).plan().units().getFirst().portablePlan();
        var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));

        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var generated = store.loadOrGenerateObserved(vectorRoute.specialization(),
                vectorRoute.kernelIr());
        Path envelope = root.resolve(vectorRoute.specialization().structuralKey() + ".artifact");
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var reloaded = store.loadOrGenerateObserved(vectorRoute.specialization(),
                vectorRoute.kernelIr());
        byte[] stale = Files.readAllBytes(envelope);
        java.nio.ByteBuffer.wrap(stale).putInt(4, 63);
        Files.write(envelope, stale);
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var recovered = store.loadOrGenerateObserved(vectorRoute.specialization(),
                vectorRoute.kernelIr());

        assertAll(
                () -> assertEquals(66, CpuGeneratorSchema.CURRENT_VERSION),
                () -> assertEquals(63, vectorRoute.specialization().classIdentitySchema()),
                () -> assertEquals(52, scalarRoute.specialization().classIdentitySchema()),
                () -> assertNotEquals(scalarRoute.specialization().structuralKey(),
                        vectorRoute.specialization().structuralKey()),
                () -> assertFalse(Arrays.equals(scalarRoute.specialization().compatibilityBytes(),
                        vectorRoute.specialization().compatibilityBytes())),
                () -> assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,
                        generated.source()),
                () -> assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT,
                        reloaded.source()),
                () -> assertArrayEquals(generated.artifact().classBytes(),
                        reloaded.artifact().classBytes()),
                () -> assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,
                        recovered.source()),
                () -> assertArrayEquals(generated.artifact().classBytes(),
                        recovered.artifact().classBytes()),
                () -> assertEquals(66, java.nio.ByteBuffer.wrap(Files.readAllBytes(envelope))
                        .getInt(4)));
    }

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
                () -> assertEquals(66, CpuGeneratorSchema.CURRENT_VERSION),
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
        byte[] schema52 = valid.clone();
        java.nio.ByteBuffer.wrap(schema52).putInt(4, 52);
        byte[] schema53 = valid.clone();
        java.nio.ByteBuffer.wrap(schema53).putInt(4, 53);
        byte[] schema54 = valid.clone();
        java.nio.ByteBuffer.wrap(schema54).putInt(4, 54);
        byte[] schema58 = valid.clone();
        java.nio.ByteBuffer.wrap(schema58).putInt(4, 58);
        byte[] malformedClass = envelope(route.specialization().structuralKey(),
                route.specialization().compatibilityBytes(), new byte[] {1, 2, 3, 4});
        byte[] wrongMetadata = envelope(route.specialization().structuralKey(),
                new byte[] {9}, seed.artifact().classBytes());
        byte[] wrongKey = envelope("0".repeat(64),
                route.specialization().compatibilityBytes(), seed.artifact().classBytes());
        for (byte[] invalid : List.of(trailing, truncated, wrongChecksum, schema58, schema54, schema53,
                schema52, schema43, schema42,
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
                    () -> assertEquals(66, java.nio.ByteBuffer.wrap(Files.readAllBytes(file))
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

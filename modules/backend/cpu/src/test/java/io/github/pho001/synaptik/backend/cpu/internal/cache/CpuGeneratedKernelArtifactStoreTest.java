package io.github.pho001.synaptik.backend.cpu.internal.cache;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGenerator;
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
import java.util.Map;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
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
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy;

class CpuGeneratedKernelArtifactStoreTest {
    @Test void softmaxBackwardReusesCurrentArtifactsAndRejectsAnotherVariantOrBakedGeometry() throws Exception {
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean log : List.of(false,true)) {
            var c = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxBackwardLoweringTest.context(type,Shape.of(2,9),log,List.of());
            var current = new CpuPartitionPreparer().analyze(c).plan().units().getFirst().portablePlan();
            var other = new CpuPartitionPreparer().analyze(io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxBackwardLoweringTest.context(type,Shape.of(2,9),!log,List.of()))
                    .plan().units().getFirst().portablePlan();
            var changed = new CpuPartitionPreparer().analyze(io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxBackwardLoweringTest.context(type,Shape.of(3,9),log,List.of()))
                    .plan().units().getFirst().portablePlan();
            var spec = current.specialization(); var generator = new CpuClassFileKernelGenerator();
            var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));
            Path path = root.resolve(spec.structuralKey()+".artifact");
            for (var stale : List.of(other,changed)) {
                var staleBytes = generator.generateClassBytes(stale.specialization(),stale.kernelIr());
                assertThrows(IllegalArgumentException.class,() -> generator.defineClassBytes(spec, current.kernelIr(), staleBytes));
                Files.write(path,envelope(spec.structuralKey(),spec.compatibilityBytes(),staleBytes));
                CpuGeneratedKernelArtifactStore.clearLoadedForTests();
                var fresh = store.loadOrGenerateObserved(spec,current.kernelIr());
                assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,fresh.source());
                assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.WEAK_INTERN_HIT,store.loadOrGenerateObserved(spec,current.kernelIr()).source());
                CpuGeneratedKernelArtifactStore.clearLoadedForTests();
                var persisted = store.loadOrGenerateObserved(spec,current.kernelIr());
                assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT,persisted.source());
                assertArrayEquals(fresh.artifact().classBytes(),persisted.artifact().classBytes());
            }
        }
    }
    /** Authentic pre-repair low payloads must miss under both stale and forged current envelopes. */
    @Test void lowFoldRejectsAuthenticPerAddClassesAndRetainsCurrentWeakAndPersistedReuse() throws Exception {
        String controls = System.getenv("SYNAPTIK_CPU0050_CONTROLS");
        org.junit.jupiter.api.Assumptions.assumeTrue(controls != null);
        for (var type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) for (boolean image : List.of(false, true)) {
            Shape output = image ? Shape.of(2, 3, 5, 7) : Shape.of(5);
            var op = image ? new Operation(io.github.pho001.synaptik.model.operation.layout.WindowTransformKind.FOLD2D,
                    new io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs(output,
                            new io.github.pho001.synaptik.model.operation.layout.Window2dAttrs(2, 3, 1, 1, 1, 2, 2, 2, false)))
                    : new Operation(io.github.pho001.synaptik.model.operation.layout.WindowTransformKind.FOLD_AXIS,
                            new io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs(0, 5, 1));
            var base = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFoldLoweringTest.context(op, type,
                    image ? Shape.of(2, 18, 35) : Shape.of(3, 3), output);
            for (int carriers = 0; carriers < 4; carriers++) {
                var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(base.partition(), base.nodes(), base.values(),
                        base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false, List.of(
                                (carriers & 1) == 0 ? CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY : CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                                (carriers & 2) == 0 ? CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY : CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)));
                if (io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertLowUnsupported(context)) continue;
                var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
                var spec = route.specialization(); var generator = new CpuClassFileKernelGenerator();
                String name = type + "-" + (image ? "2d" : "axis") + "-" + carriers;
                byte[] oldClass = Files.readAllBytes(Path.of(controls, name + ".class"));
                byte[] oldMetadata = Files.readAllBytes(Path.of(controls, name + ".compatibility"));
                byte[] oldIdentity = Files.readAllBytes(Path.of(controls, name + ".identity"));
                String oldKey = Files.readString(Path.of(controls, name + ".key"), StandardCharsets.US_ASCII);
                assertEquals(oldKey, java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(oldIdentity)));
                assertTrue(java.lang.classfile.ClassFile.of().parse(oldClass).thisClass().asInternalName().endsWith("Generated_" + oldKey));
                assertNotEquals(oldKey, spec.structuralKey());
                assertFalse(Arrays.equals(oldMetadata, spec.compatibilityBytes()));
                assertFalse(Arrays.equals(oldIdentity, spec.classIdentityBytes()));
                assertEquals(71, CpuGeneratorSchema.CURRENT_VERSION); assertEquals(52, spec.classIdentitySchema());
                assertThrows(IllegalArgumentException.class, () -> generator.defineClassBytes(spec, route.kernelIr(), oldClass));
                byte[] oldEnvelope = envelope(oldKey, oldMetadata, oldClass);
                Path oldPath = root.resolve(oldKey + ".artifact"), currentPath = root.resolve(spec.structuralKey() + ".artifact");
                Files.write(oldPath, oldEnvelope);
                var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));
                for (byte[] stale : List.of(oldEnvelope, envelope(spec.structuralKey(), oldMetadata, oldClass),
                        envelope(spec.structuralKey(), spec.compatibilityBytes(), oldClass))) {
                    Files.write(currentPath, stale); CpuGeneratedKernelArtifactStore.clearLoadedForTests();
                    var generated = store.loadOrGenerateObserved(spec, route.kernelIr());
                    assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED, generated.source());
                    assertArrayEquals(generator.generateClassBytes(spec, route.kernelIr()), generated.artifact().classBytes());
                    var weak = store.loadOrGenerateObserved(spec, route.kernelIr());
                    assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.WEAK_INTERN_HIT, weak.source());
                    assertSame(generated.artifact(), weak.artifact());
                    assertTrue(CpuGeneratedKernelArtifactStore.loadedReferencesForTests().containsKey(spec.structuralKey()));
                    assertFalse(CpuGeneratedKernelArtifactStore.loadedReferencesForTests().containsKey(oldKey));
                    CpuGeneratedKernelArtifactStore.clearLoadedForTests();
                    var persisted = store.loadOrGenerateObserved(spec, route.kernelIr());
                    assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT, persisted.source());
                    assertArrayEquals(generated.artifact().classBytes(), persisted.artifact().classBytes());
                }
                assertArrayEquals(oldEnvelope, Files.readAllBytes(oldPath));
            }
        }
    }
    /** Compact physical input has its own two-stream identity and rejects ordinary class payloads. */
    @Test void compactFoldRejectsOrdinaryPayloadAndReusesOnlyCurrentSelectedArtifacts() throws Exception {
        for(var type:List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            var context=io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuBroadcastFoldGeneratedKernelTest.context(type,
                    Shape.of(2,3,5,7),new io.github.pho001.synaptik.model.operation.layout.Window2dAttrs(2,3,1,1,0,1,1,1,false),0,false);
            var current=new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
            var oldContext=new io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionDagDecomposer().unitContext(
                    context,List.of(context.nodes().getLast()),CpuPartitionAnalysisInputs.DEFAULT);
            var ordinary=new CpuPartitionPreparer().analyze(oldContext).plan().units().getFirst().portablePlan();
            var spec=current.specialization();var generator=new CpuClassFileKernelGenerator();
            byte[] stale=generator.generateClassBytes(ordinary.specialization(),ordinary.kernelIr());
            assertFalse(Arrays.equals(ordinary.specialization().compatibilityBytes(),spec.compatibilityBytes()));
            assertFalse(Arrays.equals(ordinary.specialization().classIdentityBytes(),spec.classIdentityBytes()));
            assertThrows(IllegalArgumentException.class,()->generator.defineClassBytes(spec, current.kernelIr(), stale));
            Files.write(root.resolve(spec.structuralKey()+".artifact"),envelope(spec.structuralKey(),spec.compatibilityBytes(),stale));
            CpuGeneratedKernelArtifactStore.clearLoadedForTests();var store=new CpuGeneratedKernelArtifactStore(Optional.of(root));
            var generated=store.loadOrGenerateObserved(spec,current.kernelIr());
            assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,generated.source());
            assertSame(generated.artifact(),store.loadOrGenerate(spec,current.kernelIr()));
            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT,store.loadOrGenerateObserved(spec,current.kernelIr()).source());
        }
    }
    /** Both selective streams reject a mismatched fused topology and retain current reuse. */
    @Test void ordinaryAggregateFusionRejectsStaleTopologyAndReusesCurrentArtifact() throws Exception {
        for (var type : List.of(io.github.pho001.synaptik.model.datatype.DataType.FLOAT32,io.github.pho001.synaptik.model.datatype.DataType.FLOAT64)) {
            var old = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAggregateFusionGeneratedKernelTest.ordinaryRoute(type,0);
            var current = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAggregateFusionGeneratedKernelTest.ordinaryRoute(type,2);
            var spec = current.specialization(); var generator = new CpuClassFileKernelGenerator();
            byte[] stale = generator.generateClassBytes(old.specialization(),old.kernelIr());
            assertFalse(Arrays.equals(old.specialization().compatibilityBytes(),spec.compatibilityBytes()));
            assertFalse(Arrays.equals(old.specialization().classIdentityBytes(),spec.classIdentityBytes()));
            assertThrows(IllegalArgumentException.class,() -> generator.defineClassBytes(spec, current.kernelIr(), stale));
            Files.write(root.resolve(spec.structuralKey()+".artifact"),envelope(spec.structuralKey(),old.specialization().compatibilityBytes(),stale));
            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));
            var generated = store.loadOrGenerateObserved(spec,current.kernelIr());
            assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,generated.source());
            assertSame(generated.artifact(),store.loadOrGenerate(spec,current.kernelIr()));
            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            var persisted = store.loadOrGenerateObserved(spec,current.kernelIr());
            assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT,persisted.source());
            assertArrayEquals(generated.artifact().classBytes(),persisted.artifact().classBytes());
        }
    }
    @TempDir Path root;

    /**
     * Rejects genuine frozen CPU0073 scalar bytes under the current production-selected
     * controlled identity, then verifies generation, weak reuse and persistent reload.
     * Frozen bytes are authenticated independently of the changed fresh derivative.
     * @throws Exception if payload decoding, authentication or persistence fails
     */
    @Test void controlledScanRejectsFrozenScalarShapesAndReusesExactProductionArtifact() throws Exception {
        var fixture=io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuRowMapReductionGeneratedKernelTest.class;
        for(var type:List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            byte[] before;
            try(var input=fixture.getResourceAsStream("row-fusion-cpu0073-before-"+type+".class.gz.b64")) {
                assertNotNull(input);
                try(var gzip=new java.util.zip.GZIPInputStream(new java.io.ByteArrayInputStream(java.util.Base64.getMimeDecoder().decode(input.readAllBytes())))){before=gzip.readAllBytes();}
            }
            assertEquals(type==DataType.FLOAT32?"486a51c6a415bf5a48b084dc8b5ffad030a070c1851e7aa18d1ef5ed1d0072d8":"e3a46f8a7b886a3ac2bd6d92565bdf2dcf7b0082ee32b56951ffc4bddb26a715",
                    java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(before)));
            var compiled=io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuMatmulBindingSnapshotTest.conditionalFamily(type,true);
            var analysis=new CpuPartitionPreparer().analyze(io.github.pho001.synaptik.prepare.GraphPreparation.project(
                    compiled,compiled.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT));
            var route=analysis.plan().units().stream().filter(unit -> unit.runtimeFacts().generalScan().isPresent()).toList().getLast().portablePlan();
            var spec=route.specialization(); var ir=route.kernelIr(); var generator=new CpuClassFileKernelGenerator();
            assertThrows(IllegalArgumentException.class,()->generator.defineClassBytes(spec,ir,before));
            Files.write(root.resolve(spec.structuralKey()+".artifact"),envelope(spec.structuralKey(),spec.compatibilityBytes(),before));
            CpuGeneratedKernelArtifactStore.clearLoadedForTests(); var store=new CpuGeneratedKernelArtifactStore(Optional.of(root));
            var generated=store.loadOrGenerateObserved(spec,ir);
            assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,generated.source());
            assertSame(generated.artifact(),store.loadOrGenerate(spec,ir));
            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            var reloaded=store.loadOrGenerateObserved(spec,ir);
            assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT,reloaded.source());
            assertArrayEquals(generated.artifact().classBytes(),reloaded.artifact().classBytes());
        }
    }

    /**
     * Retires literal eligible pre-edit ClassFiles and both captured streams, while retaining
     * current weak/persisted reuse. Fixtures were frozen before any production edit; no old
     * expected identity is reconstructed by stripping a candidate suffix.
     * @throws Exception if immutable payload decoding, envelope checks or cache persistence fails
     */
    @Test void rowRegionsRejectLiteralOldArtifactsAndReuseCurrentArtifacts() throws Exception {
        var fixture = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuRowMapReductionGeneratedKernelTest.class;
        var frozen = new java.util.HashMap<String,String>();
        try (var stream = fixture.getResourceAsStream("row-map-reduction-before.tsv")) {
            assertNotNull(stream);
            new String(stream.readAllBytes(),StandardCharsets.UTF_8).lines().filter(l -> !l.startsWith("#") && !l.isBlank())
                    .map(l -> l.split("\t")).forEach(row -> frozen.put(row[0],row[1]));
        }
        for (var type : List.of(DataType.FLOAT32,DataType.FLOAT64)) {
            var old = new java.util.HashMap<String,byte[]>();
            for (String extension : List.of("class","compatibility","classidentity")) {
                try (var stream = fixture.getResourceAsStream("row-map-before-"+type+"."+extension+".gz.b64")) {
                    assertNotNull(stream);
                    try (var gzip = new java.util.zip.GZIPInputStream(new java.io.ByteArrayInputStream(java.util.Base64.getMimeDecoder().decode(stream.readAllBytes())))) {
                        byte[] payload = gzip.readAllBytes(); old.put(extension,payload);
                        assertEquals(frozen.get("combined-"+type+"-1."+extension),java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(payload)));
                    }
                }
            }
            var compiled=io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuMatmulBindingSnapshotTest.conditionalFamily(type,true);
            var analysis=new CpuPartitionPreparer().analyze(io.github.pho001.synaptik.prepare.GraphPreparation.project(
                    compiled,compiled.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT));
            var route=analysis.plan().units().stream().filter(unit -> unit.runtimeFacts().generalScan().isPresent()).toList().getLast().portablePlan();
            var spec=route.specialization(); var ir=route.kernelIr();
            String oldKey = java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(old.get("classidentity")));
            assertNotEquals(oldKey,spec.structuralKey());assertFalse(Arrays.equals(old.get("compatibility"),spec.compatibilityBytes()));
            var generator = new CpuClassFileKernelGenerator();
            assertThrows(IllegalArgumentException.class,() -> generator.defineClassBytes(spec, ir, old.get("class")));
            var path = root.resolve(spec.structuralKey()+".artifact");
            var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));
            for (byte[] stale : List.of(envelope(oldKey,old.get("compatibility"),old.get("class")),
                    envelope(spec.structuralKey(),old.get("compatibility"),old.get("class")),
                    envelope(spec.structuralKey(),spec.compatibilityBytes(),old.get("class")))) {
                Files.write(path,stale);CpuGeneratedKernelArtifactStore.clearLoadedForTests();
                var generated = store.loadOrGenerateObserved(spec,ir);
                assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,generated.source());
                assertArrayEquals(generator.generateClassBytes(spec,ir),generated.artifact().classBytes());
                assertSame(generated.artifact(),store.loadOrGenerate(spec,ir));
            }
            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT,store.loadOrGenerateObserved(spec,ir).source());
        }
    }

    /**
     * Models collection through the exact weak token instead of depending on GC timing; a
     * different-key cold realization must drain it while retaining an independently live entry.
     */
    @Test void coldRealizationReclaimsCollectedKeysButKeepsLiveArtifacts() {
        var firstRoute = CpuPartitionPreparerTest.analyze(Shape.of(8))
                .plan().units().getFirst().portablePlan();
        var shape = Shape.of(8);
        var descriptor = new io.github.pho001.synaptik.model.tensor.TensorDescriptor(
                DataType.FLOAT32, shape,
                Optional.of(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(shape)),
                false);
        var otherRoute = CpuPartitionPreparerTest.analyze(descriptor, descriptor, descriptor,
                descriptor, CpuPartitionAnalysisInputs.DEFAULT).plan().units().getFirst().portablePlan();
        String firstKey = firstRoute.specialization().structuralKey();
        String otherKey = otherRoute.specialization().structuralKey();
        assertNotEquals(firstKey, otherKey);
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var store = new CpuGeneratedKernelArtifactStore();
        var first = store.loadOrGenerate(firstRoute.specialization(), firstRoute.kernelIr());
        var firstReference = CpuGeneratedKernelArtifactStore.loadedReferencesForTests().get(firstKey);
        assertSame(first, firstReference.get());
        firstReference.clear();
        assertTrue(firstReference.enqueue());
        assertTrue(CpuGeneratedKernelArtifactStore.loadedReferencesForTests().containsKey(firstKey));

        var live = store.loadOrGenerate(otherRoute.specialization(), otherRoute.kernelIr());
        assertEquals(java.util.Set.of(otherKey),
                CpuGeneratedKernelArtifactStore.loadedReferencesForTests().keySet());
        var reused = new CpuGeneratedKernelArtifactStore().loadOrGenerateObserved(
                otherRoute.specialization(), otherRoute.kernelIr());
        assertSame(live, reused.artifact());
        assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.WEAK_INTERN_HIT,
                reused.source());
    }

    /**
     * Models queue delivery after replacement, both normally and across the isolated-test reset.
     * Strong test locals deliberately keep artifacts available for exact identity assertions.
     */
    @Test void delayedCollectionCannotRemoveANewerSameKeyArtifact() {
        var route = CpuPartitionPreparerTest.analyze(Shape.of(8))
                .plan().units().getFirst().portablePlan();
        String key = route.specialization().structuralKey();
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var store = new CpuGeneratedKernelArtifactStore();
        var first = store.loadOrGenerate(route.specialization(), route.kernelIr());
        var obsolete = CpuGeneratedKernelArtifactStore.loadedReferencesForTests().get(key);
        assertSame(first, obsolete.get());
        obsolete.clear();
        var replacement = store.loadOrGenerate(route.specialization(), route.kernelIr());
        assertNotSame(first, replacement);
        assertTrue(obsolete.enqueue());
        var reused = store.loadOrGenerateObserved(route.specialization(), route.kernelIr());
        assertSame(replacement, reused.artifact());
        assertSame(replacement,
                CpuGeneratedKernelArtifactStore.loadedReferencesForTests().get(key).get());
        assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.WEAK_INTERN_HIT,
                reused.source());

        var beforeReset = CpuGeneratedKernelArtifactStore.loadedReferencesForTests().get(key);
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var afterReset = store.loadOrGenerate(route.specialization(), route.kernelIr());
        beforeReset.clear();
        assertTrue(beforeReset.enqueue());
        assertSame(afterReset, store.loadOrGenerate(route.specialization(), route.kernelIr()));
    }

    /**
     * Releases four callers together and checks the existing atomic publication/reuse contract.
     *
     * @throws Exception if bounded worker synchronization or artifact realization fails
     */
    @Test void concurrentRealizationsPublishOneLiveArtifact() throws Exception {
        var route = CpuPartitionPreparerTest.analyze(Shape.of(8))
                .plan().units().getFirst().portablePlan();
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var start = new CyclicBarrier(4);
        var futures = new java.util.ArrayList<
                java.util.concurrent.Future<CpuGeneratedKernelArtifactStore.ObservedRealization>>();
        try (var executor = Executors.newFixedThreadPool(4)) {
            for (int thread = 0; thread < 4; thread++) {
                futures.add(executor.submit(() -> {
                    start.await(10, TimeUnit.SECONDS);
                    return new CpuGeneratedKernelArtifactStore().loadOrGenerateObserved(
                            route.specialization(), route.kernelIr());
                }));
            }
            var first = futures.getFirst().get(10, TimeUnit.SECONDS).artifact();
            int generated = 0;
            for (var future : futures) {
                var result = future.get(10, TimeUnit.SECONDS);
                assertSame(first, result.artifact());
                if (result.source() == CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED) {
                    generated++;
                } else {
                    assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.WEAK_INTERN_HIT,
                            result.source());
                }
            }
            assertEquals(1, generated);
        }
    }

    /**
     * Authenticates typed outer/combined namespaces and real ClassFiles, rejecting combined
     * baked v1..v8 metadata (including withdrawn v8) while retaining canonical v1 and typed outer.
     * @throws Exception if the isolated artifact files cannot be read or written
     */
    @Test void typedOrdinaryAndCombinedLengthArtifactsReuseAndRejectStaleOrForeignPayloads() throws Exception {
        var b = new io.github.pho001.synaptik.model.shape.DynamicDimension("B");
        var row = Shape.ofDimensions(b,new io.github.pho001.synaptik.model.shape.StaticDimension(3));
        var source = new io.github.pho001.synaptik.model.tensor.TensorDescriptor(DataType.FLOAT64,row,Optional.empty(),false);
        var scalar = new io.github.pho001.synaptik.model.tensor.TensorDescriptor(DataType.FLOAT64,Shape.scalar(),Optional.empty(),false);
        var context = CpuNonAffineMovementLoweringTest.context(new Operation(io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.MEAN,
                io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE),List.of(0),List.of(source),scalar);
        var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
        var facts = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering.analyze(
                io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.symbolicLengthQuery(DataType.FLOAT64,4,false));
        var spec = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.specialization(facts);
        var entries = List.of(Map.entry(route.specialization(),route.kernelIr()),Map.entry(spec,facts.ir().encodedKernelIr()));
        var generator = new CpuClassFileKernelGenerator(); var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));
        for (var candidate : entries) {
            var current = candidate.getKey(); var ir = candidate.getValue(); var path = root.resolve(current.structuralKey()+".artifact");
            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            var first = store.loadOrGenerateObserved(current,ir); assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,first.source());
            assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.WEAK_INTERN_HIT,store.loadOrGenerateObserved(current,ir).source());
            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT,store.loadOrGenerateObserved(current,ir).source());
            byte[] expected = generator.generateClassBytes(current,ir);
            String old = new String(current.compatibilityBytes(),StandardCharsets.US_ASCII)
                    .replace("ordinary-batch-typed-v1","ordinary-batch-v1").replace("scan-symbolic-length-v9","scan-symbolic-length-v1");
            assertFalse(Arrays.equals(old.getBytes(StandardCharsets.US_ASCII),current.compatibilityBytes()));
            byte[] foreign = generator.generateClassBytes(entries.get(candidate == entries.getFirst() ? 1 : 0).getKey(),entries.get(candidate == entries.getFirst() ? 1 : 0).getValue());
            assertThrows(IllegalArgumentException.class,() -> generator.defineClassBytes(current, ir, foreign));
            for (byte[] stale : List.of(envelope(current.structuralKey(),old.getBytes(StandardCharsets.US_ASCII),expected),
                    envelope(current.structuralKey(),current.compatibilityBytes(),foreign))) {
                Files.write(path,stale); CpuGeneratedKernelArtifactStore.clearLoadedForTests();
                var recovered = store.loadOrGenerateObserved(current,ir);
                assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,recovered.source()); assertArrayEquals(expected,recovered.artifact().classBytes());
            }
            if (current.generalScanIr().isPresent()) for (int version : new int[]{1,2,3,4,5,6,7,8}) {
                String stale = new String(current.compatibilityBytes(),StandardCharsets.US_ASCII).replace("|scan-symbolic-length-v9=","|scan-symbolic-length-v"+version+"=");
                assertFalse(Arrays.equals(stale.getBytes(StandardCharsets.US_ASCII),current.compatibilityBytes()));
                Files.write(path,envelope(current.structuralKey(),stale.getBytes(StandardCharsets.US_ASCII),expected));
                CpuGeneratedKernelArtifactStore.clearLoadedForTests();
                var recovered = store.loadOrGenerateObserved(current,ir);
                assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,recovered.source(),"combined v"+version+" metadata is stale");
                assertArrayEquals(expected,recovered.artifact().classBytes());
            }
            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT,store.loadOrGenerateObserved(current,ir).source(),"recovered current artifact reuses");
            assertEquals(71,CpuGeneratorSchema.CURRENT_VERSION);
        }
    }

    /**
     * Rejects real accepted v1 Class-Files at definition and persistence seams. These compressed
     * fixtures were generated from the pre-CPU0034 accepted source archive, not by rewriting new
     * code. Their externally recorded CPU0033 hashes and v1 identity streams remain provenance.
     */
    @Test void stagedV4RejectsActualAcceptedV1ClassesAndBothMetadataStreamsThenReusesCompatibleTime() throws Exception {
        var cases = List.of(
                io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.shapedFacts(3,2,3),
                io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.matmulFacts(3,2,3,4));
        var compatible = List.of(
                io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.shapedFacts(257,2,3),
                io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.matmulFacts(257,2,3,4));
        var oldHashes = List.of("8571e6a50f3498a716d7a2ae9d9715b6d5d2a97a3b2895efe3a451f57a8a0832",
                "89f788e4cd4be04597c1ecb31644bf329c9d039ddf6fb30ca3069bcbe8ca9895");
        var oldKeys = List.of("03e3dc889a43f6b8941557b7570128672d35bf841a27f849eae36a01971115f1",
                "e1de5745d528dec19f42d45358d25bc1854bac85c5f2b794284c243e2cda5b58");
        var generator = new CpuClassFileKernelGenerator(); var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));
        for (int i = 0; i < cases.size(); i++) {
            var facts = cases.get(i);
            var spec = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.specialization(facts);
            String family = i == 0 ? "shaped" : "matmul";
            byte[] oldBytes;
            try (var resource = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.class
                    .getResourceAsStream("general-scan-v1-"+family+".class.gz.b64")) {
                assertNotNull(resource);
                try (var gzip = new java.util.zip.GZIPInputStream(new java.io.ByteArrayInputStream(
                        java.util.Base64.getMimeDecoder().decode(resource.readAllBytes())))) { oldBytes = gzip.readAllBytes(); }
            }
            assertEquals(oldHashes.get(i),io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.sha256(oldBytes));
            assertTrue(java.lang.classfile.ClassFile.of().verify(oldBytes).isEmpty());
            assertEquals(oldKeys.get(i),io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.sha256(
                    io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.v1Identity(spec.classIdentityBytes())));
            assertThrows(IllegalArgumentException.class,() -> generator.defineClassBytes(spec, spec.generalScanIr().orElseThrow().encodedKernelIr(), oldBytes));
            byte[] oldMetadata = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.v1Identity(spec.compatibilityBytes());
            assertFalse(Arrays.equals(oldMetadata,spec.compatibilityBytes()));
            Path current = root.resolve(spec.structuralKey()+".artifact"), old = root.resolve(oldKeys.get(i)+".artifact");
            Files.write(old,envelope(oldKeys.get(i),oldMetadata,oldBytes));
            byte[] oldEnvelope = Files.readAllBytes(old);
            // First prove normal old-key isolation, then defeat path isolation to exercise each
            // real same-key validation seam with checksummed, structurally valid envelopes.
            for (byte[] stale : List.of(oldEnvelope,envelope(spec.structuralKey(),oldMetadata,oldBytes),
                    envelope(spec.structuralKey(),spec.compatibilityBytes(),oldBytes))) {
                Files.write(current,stale); CpuGeneratedKernelArtifactStore.clearLoadedForTests();
                var result = store.loadOrGenerateObserved(spec,facts.ir().encodedKernelIr());
                assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,result.source());
                assertArrayEquals(generator.generateClassBytes(spec,facts.ir().encodedKernelIr()),result.artifact().classBytes());
            }
            assertArrayEquals(oldEnvelope,Files.readAllBytes(old),"stale old-key entry is never mutated");
            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            var same = compatible.get(i);
            var reused = store.loadOrGenerateObserved(io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.specialization(same),same.ir().encodedKernelIr());
            assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT,reused.source());
            assertEquals(71,CpuGeneratorSchema.CURRENT_VERSION); assertEquals(71,spec.classIdentitySchema());
        }
    }

    /**
     * Rejects actual CPU0034/CPU0035 artifacts with their original geometry/dtype provenance.
     * @param version historical staged payload version, two or three
     * @throws Exception if fixture decoding, identity checks or persisted-artifact access fails
     */
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(ints={2,3})
    void stagedV4RejectsVerifiedActualV2AndV3ClassesWithoutMutatingOldKeysAndReusesRuntimeTime(int version) throws Exception {
        var cases = List.of(
                io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.broadcastFacts(DataType.FLOAT64,8,4,8,false,true),
                io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.matmulFacts(4,2,3,4));
        var compatible = List.of(
                io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.broadcastFacts(DataType.FLOAT64,257,4,8,false,true),
                io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.matmulFacts(257,2,3,4));
        var hashes = version == 2 ? List.of("32881953398f155906c8e4cc554ccaf60e14fb80c1c116872cb273465c4e75bb",
                "55dbf382568dcfa7ae4317776346ca37752604ffacdeb88c1c60f7b0ad702a6a")
                : List.of("870c954f4d83597980d5ea9d5c704a89c9cd8a459fbf35ca85cadc14ce01014e",
                "4f1d9e33375de8b941488920e86de18045dafadda2fb53e084e026bd708ccb1d");
        var keys = version == 2 ? List.of("9bc5185e846023927c0c65be4fccef83fc1ef19f2503908f2b6ca17728b850da",
                "3855fc1972754b9ddaca278bec16176bad6a97b355eeaafd43d5c173145119c5")
                : List.of("295312b55b6f922dc77232e71af1d6dc22e49de755266b8bc15bd542d23556af",
                "c4c9578b22b5ed9442d38fe12f50e5aea560a9a03fe39eb8ce71b9576b083521");
        var metadataHashes = version == 2 ? List.of("e05ad4ce7142f796bd83a367557ad8ff9be0f7b8c5ccdbba8614cf89aff2e085",
                "8ea1e69f6a0f15a79016583c03c3a9038ca10706b457b6cdd55637114b8b5108")
                : List.of("fb962fd1ccdd6188f3bb90cbf349c5e45c87b71a948041b90b439fe55b8aeb5d",
                "91e9cec27d2c202aa129c7ece9e81d2b087df9a2a906e130ccff783953ccf393");
        var generator = new CpuClassFileKernelGenerator(); var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));
        for (int i = 0; i < cases.size(); i++) {
            var facts = cases.get(i);
            var spec = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.specialization(facts);
            String family = i == 0 ? "shaped" : "matmul";
            byte[] oldBytes;
            try (var resource = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.class
                    .getResourceAsStream("general-scan-v"+version+"-"+family+".class.gz.b64")) {
                assertNotNull(resource);
                try (var gzip = new java.util.zip.GZIPInputStream(new java.io.ByteArrayInputStream(
                        java.util.Base64.getMimeDecoder().decode(resource.readAllBytes())))) { oldBytes = gzip.readAllBytes(); }
            }
            assertEquals(hashes.get(i),io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.sha256(oldBytes));
            var model = java.lang.classfile.ClassFile.of().parse(oldBytes);
            assertTrue(java.lang.classfile.ClassFile.of().verify(oldBytes).isEmpty());
            assertTrue(model.thisClass().asInternalName().endsWith("Generated_"+keys.get(i)),"actual recorded binary identity, not patched bytecode");
            byte[] identity = version == 2
                    ? io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.v2Identity(spec.classIdentityBytes())
                    : io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.v3Identity(spec.classIdentityBytes());
            assertEquals(keys.get(i),io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.sha256(identity));
            byte[] metadata = version == 2
                    ? io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.v2Identity(spec.compatibilityBytes())
                    : io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.v3Identity(spec.compatibilityBytes());
            assertEquals(metadataHashes.get(i),io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.sha256(metadata));
            assertThrows(IllegalArgumentException.class,() -> generator.defineClassBytes(spec, spec.generalScanIr().orElseThrow().encodedKernelIr(), oldBytes));
            Path old = root.resolve(keys.get(i)+".artifact"), current = root.resolve(spec.structuralKey()+".artifact");
            byte[] oldEnvelope = envelope(keys.get(i),metadata,oldBytes); Files.write(old,oldEnvelope);
            for (byte[] stale : List.of(oldEnvelope,envelope(spec.structuralKey(),metadata,oldBytes),
                    envelope(spec.structuralKey(),spec.compatibilityBytes(),oldBytes))) {
                Files.write(current,stale); CpuGeneratedKernelArtifactStore.clearLoadedForTests();
                var generated = store.loadOrGenerateObserved(spec,facts.ir().encodedKernelIr());
                assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,generated.source());
                assertArrayEquals(generator.generateClassBytes(spec,facts.ir().encodedKernelIr()),generated.artifact().classBytes());
            }
            assertArrayEquals(oldEnvelope,Files.readAllBytes(old));
            CpuGeneratedKernelArtifactStore.clearLoadedForTests();
            var same = compatible.get(i); var sameSpec = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.specialization(same);
            assertEquals(spec,sameSpec); assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT,
                    store.loadOrGenerateObserved(sameSpec,same.ir().encodedKernelIr()).source());
            assertEquals(71,CpuGeneratorSchema.CURRENT_VERSION); assertEquals(71,spec.classIdentitySchema());
        }
    }

    @Test void shapedScanSeparatesBakedGeometryInWeakAndPersistedStoresWithoutEnvelopeChurn() throws Exception {
        var a = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.shapedFacts(3,2,3);
        var b = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.shapedFacts(3,4,7);
        var sa = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.specialization(a);
        var sb = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanGeneratedKernelTest.specialization(b);
        assertEquals(a.ir().encodedKernelIr(),b.ir().encodedKernelIr());
        var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var aa = store.loadOrGenerateObserved(sa,a.ir().encodedKernelIr());
        var bb = store.loadOrGenerateObserved(sb,b.ir().encodedKernelIr());
        assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,bb.source());
        assertNotSame(aa.artifact(),bb.artifact());
        assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.WEAK_INTERN_HIT,
                store.loadOrGenerateObserved(sa,a.ir().encodedKernelIr()).source());
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT,
                store.loadOrGenerateObserved(sa,a.ir().encodedKernelIr()).source());
        // A valid current-schema artifact for another baked geometry must not be accepted even
        // when copied to the requested key's path. Preserve historical fixture guards unchanged.
        Files.copy(root.resolve(sa.structuralKey()+".artifact"),root.resolve(sb.structuralKey()+".artifact"),
                java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,
                store.loadOrGenerateObserved(sb,b.ir().encodedKernelIr()).source());
        assertEquals(71,CpuGeneratorSchema.CURRENT_VERSION);
    }

    /** Low GELU remains a real BFLOAT16 cache-admission query, not an F32 projection. */
    @Test void lowActivationRejectsFreshArtifactAdmission() {
        var read = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ, CpuAccessPlan.Regime.DENSE_LINEAR,
                1, List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var write = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE, CpuAccessPlan.Regime.DENSE_LINEAR,
                1, List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var ir = new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, DataType.BFLOAT16, CpuKernelIr.Value.Kind.INPUT, read),
                new CpuKernelIr.Value(1, DataType.BFLOAT16, CpuKernelIr.Value.Kind.OUTPUT, write)),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.GELU_EXACT, List.of(0), 1)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(1, 0)));
        var specialization = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                ExecutionStrategy.SCALAR, List.of(DataType.BFLOAT16, DataType.BFLOAT16),
                List.of(CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY,
                        CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY), 0, -1, List.of(), false, 59);
        var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        assertThrows(IllegalArgumentException.class,
                () -> store.loadOrGenerateObserved(specialization, ir));
        assertFalse(Files.exists(root.resolve(specialization.structuralKey() + ".artifact")));
        assertFalse(CpuGeneratedKernelArtifactStore.loadedReferencesForTests()
                .containsKey(specialization.structuralKey()));
    }

    /**
     * Checks envelope-version invalidation independently on an eligible FLOAT64 activation.
     * The changed header is deliberately not claimed to be an authentic schema-70 ClassFile.
     * @throws Exception if artifact persistence or reading fails
     */
    @Test void eligibleActivationRejectsStaleEnvelopeHeaderThenReusesCurrentArtifact() throws Exception {
        var descriptor = new io.github.pho001.synaptik.model.tensor.TensorDescriptor(
                DataType.FLOAT64, Shape.of(3), Optional.of(
                        io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(Shape.of(3))),
                false);
        var context = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest
                .context(new Operation(
                        io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind.GELU,
                        io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE),
                        List.of(0), List.of(descriptor), descriptor);
        var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
        var specialization = route.specialization();
        var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var initial = store.loadOrGenerateObserved(specialization, route.kernelIr());
        Path file = root.resolve(specialization.structuralKey() + ".artifact");
        byte[] current = Files.readAllBytes(file);
        byte[] stale = current.clone();
        java.nio.ByteBuffer.wrap(stale).putInt(4, 70);
        Files.write(file, stale);
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var regenerated = store.loadOrGenerateObserved(specialization, route.kernelIr());
        assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED, regenerated.source());
        assertArrayEquals(initial.artifact().classBytes(), regenerated.artifact().classBytes());
        assertArrayEquals(current, Files.readAllBytes(file));
        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var persisted = store.loadOrGenerateObserved(specialization, route.kernelIr());
        assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT, persisted.source());
        assertArrayEquals(initial.artifact().classBytes(), persisted.artifact().classBytes());
    }

    @Test void aggregateSegmentArtifactRejectsSchema68EnvelopeAndRegenerates() throws Exception {
        var base = io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest
                .context(io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.SUM,
                        DataType.FLOAT32, Shape.of(2, 3),
                        new io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs(1,
                                false), Shape.of(2));
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false,
                List.of(CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                CpuPartitionAnalysisInputs.DEFAULT.portableExecution()));
        var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
        var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));
        Path envelope = root.resolve(route.specialization().structuralKey() + ".artifact");

        CpuGeneratedKernelArtifactStore.clearLoadedForTests();
        var initial = store.loadOrGenerateObserved(route.specialization(), route.kernelIr());
        byte[] stale = Files.readAllBytes(envelope);
        java.nio.ByteBuffer.wrap(stale).putInt(4, 68);
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
                () -> assertEquals(71, java.nio.ByteBuffer.wrap(Files.readAllBytes(envelope))
                        .getInt(4)));
    }

    /**
     * Proves current-only envelope invalidation against admitted composition routes, rather than
     * a hand-written compatibility payload.
     */
    @Test void concatAndStackRejectSameKeySchema64EnvelopesThenPersistSchema71Hits() throws Exception {
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
            assertEquals(71, java.nio.ByteBuffer.wrap(Files.readAllBytes(file)).getInt(4),
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

    @Test void publishesReloadsSchema71ConvAndRejectsStaleSchema64Envelope() throws Exception {
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
        var vectorContext = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), base.constants(), vectorInputs);
        var scalarContext = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(base.partition(), base.nodes(), base.values(), base.memoryRequirements(), base.constants(), scalarInputs);
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
                () -> assertEquals(71, CpuGeneratorSchema.CURRENT_VERSION),
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
                () -> assertEquals(71, java.nio.ByteBuffer.wrap(Files.readAllBytes(envelope))
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
                () -> assertEquals(71, CpuGeneratorSchema.CURRENT_VERSION),
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
        var vectorContext = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(descriptor.partition(), descriptor.nodes(), descriptor.values(), descriptor.memoryRequirements(), descriptor.constants(), vectorInputs);
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
                    () -> assertEquals(71, java.nio.ByteBuffer.wrap(Files.readAllBytes(file))
                            .getInt(4)),
                    () -> assertTrue(Files.size(file) <=
                            CpuGeneratedKernelArtifactStore.MAX_ENVELOPE_BYTES));
        }
    }

    /**
     * Rejects authenticated pre-CPU0071 static classes under old keys and forged current
     * envelopes, then proves exact reuse of artifacts selected by current production analysis.
     * @throws Exception if immutable fixture decoding, cache validation or persistence fails
     */
    @Test void cpu0071StaticMatmulCutoverRejectsOldClassesAndReusesCurrentArtifacts() throws Exception {
        var captured = new java.util.HashMap<String,String[]>();
        try (var stream = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuMatmulBindingSnapshotTest.class
                .getResourceAsStream("matmul-binding-before.tsv")) {
            assertNotNull(stream);
            new String(stream.readAllBytes(),StandardCharsets.UTF_8).lines().filter(l -> !l.startsWith("#") && !l.isBlank())
                    .map(l -> l.split("\t")).forEach(row -> captured.put(row[0],row));
        }
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean length : List.of(false,true)) {
            var compiled = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuMatmulBindingSnapshotTest.conditionalFamily(type,length);
            var analysis = new CpuPartitionPreparer().analyze(io.github.pho001.synaptik.prepare.GraphPreparation.project(
                    compiled,compiled.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT));
            int phase = 0;
            for (var unit : analysis.plan().units()) {
                if (unit.runtimeFacts().generalScan().isEmpty()) continue;
                var route = unit.portablePlan(); var spec = route.specialization(); var ir = route.kernelIr();
                String name = "conditional-"+type+"-"+length+"-"+phase++;
                byte[] oldBytes;
                try (var input = io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuMatmulBindingSnapshotTest.class
                        .getResourceAsStream("matmul-binding-before-"+name+".class.gz.b64")) {
                    assertNotNull(input);
                    try (var gzip = new java.util.zip.GZIPInputStream(new java.io.ByteArrayInputStream(java.util.Base64.getMimeDecoder().decode(input.readAllBytes())))) { oldBytes = gzip.readAllBytes(); }
                }
                var recorded = captured.get(name); assertNotNull(recorded);
                assertEquals(recorded[1],java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(oldBytes)));
                String oldKey = recorded[3];
                assertTrue(java.lang.classfile.ClassFile.of().parse(oldBytes).thisClass().asInternalName().endsWith("Generated_"+oldKey),
                        "authentic frozen binary identity, independent of the changed current derivative");
                assertNotEquals(oldKey,spec.structuralKey());
                var generator = new CpuClassFileKernelGenerator();
                assertThrows(IllegalArgumentException.class,() -> generator.defineClassBytes(spec,ir,oldBytes));
                Path oldPath = root.resolve(oldKey+".artifact"), currentPath = root.resolve(spec.structuralKey()+".artifact");
                // Deliberately forged metadata is not a reconstruction of a historical stream.
                byte[] oldEnvelope = envelope(oldKey,new byte[]{0},oldBytes); Files.write(oldPath,oldEnvelope);
                var store = new CpuGeneratedKernelArtifactStore(Optional.of(root));
                for (byte[] stale : List.of(oldEnvelope,envelope(spec.structuralKey(),new byte[]{0},oldBytes),
                        envelope(spec.structuralKey(),spec.compatibilityBytes(),oldBytes))) {
                    Files.write(currentPath,stale); CpuGeneratedKernelArtifactStore.clearLoadedForTests();
                    var result = store.loadOrGenerateObserved(spec,ir);
                    assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.GENERATED,result.source());
                    assertArrayEquals(generator.generateClassBytes(spec,ir),result.artifact().classBytes());
                }
                assertArrayEquals(oldEnvelope,Files.readAllBytes(oldPath),"old key remains untouched");
                CpuGeneratedKernelArtifactStore.clearLoadedForTests();
                assertEquals(CpuGeneratedKernelArtifactStore.RealizationSource.PERSISTED_HIT,store.loadOrGenerateObserved(spec,ir).source());
            }
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

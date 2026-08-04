package io.github.pho001.synaptik.backend.cpu.execution;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.lang.classfile.ClassFile;
import java.lang.ref.Reference;
import java.lang.reflect.Modifier;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CpuGeneratedKernelArtifactStoreTest {
    private static final long TIMEOUT_SECONDS = 10;
    private static final CpuLoweringFingerprint LOWERING =
            CpuLoweringFingerprint.of(new byte[] {3, 1, 4, 1, 5});

    @TempDir Path temporaryDirectory;

    @Test void validatesSurfaceRootAndInputsInStableOrder() throws Exception {
        assertAll(
                () -> assertEquals("root", assertThrows(NullPointerException.class,
                        () -> new CpuGeneratedKernelArtifactStore(null)).getMessage()),
                () -> assertFalse(Modifier.isPublic(
                        CpuGeneratedKernelArtifactStore.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(
                        CpuGeneratedKernelArtifactStore.class.getModifiers())),
                () -> assertEquals(1,
                        CpuGeneratedKernelArtifactStore.class.getDeclaredConstructors().length));
        long operations = java.util.Arrays.stream(
                        CpuGeneratedKernelArtifactStore.class.getDeclaredMethods())
                .filter(method -> !Modifier.isPrivate(method.getModifiers()))
                .filter(method -> !method.isSynthetic())
                .count();
        assertEquals(1, operations);

        var store = new CpuGeneratedKernelArtifactStore(temporaryDirectory.resolve("a/../root"));
        CpuKernelSpecialization specialization = specialization(1);
        assertAll(
                () -> assertEquals("specialization", assertThrows(NullPointerException.class,
                        () -> store.loadOrGenerate(null, null)).getMessage()),
                () -> assertEquals("familyEmitter", assertThrows(NullPointerException.class,
                        () -> store.loadOrGenerate(specialization, null)).getMessage()),
                () -> assertEquals(
                        "familyEmitter lowering fingerprint does not match specialization",
                        assertThrows(IllegalArgumentException.class,
                                () -> store.loadOrGenerate(specialization,
                                        new EmptyEmitter(CpuLoweringFingerprint.of(
                                                new byte[] {9}), null))).getMessage()));
        assertFalse(Files.exists(temporaryDirectory.resolve("root")));
    }

    @Test void publishesExactEnvelopePathAndReusesDiskAcrossStoreInstances() throws Throwable {
        Path root = temporaryDirectory.resolve("root");
        CpuKernelSpecialization specialization = specialization(1);
        var emissions = new AtomicInteger();
        CpuGeneratedKernel first = new CpuGeneratedKernelArtifactStore(root).loadOrGenerate(
                specialization, new EmptyEmitter(LOWERING, emissions));
        Path entry = onlyEntry(root);
        byte[] bytes = Files.readAllBytes(entry);
        byte[] magic = CpuGeneratorSchema.ARTIFACT_MAGIC.getBytes(
                java.nio.charset.StandardCharsets.US_ASCII);
        assertArrayEquals(magic, java.util.Arrays.copyOf(bytes, magic.length));
        assertTrue(entry.toString().replace('\\', '/').matches(
                ".*/generated-kernels/v1/sha256/[0-9a-f]{2}/[0-9a-f]{62}\\.cpuclass"));
        assertEquals("generated-kernels/v1/sha256/38/"
                        + "789fb4906c39c24843ca66625bde7adbbd7ab1d848cf86ddf11e73e79f1e3e.cpuclass",
                root.relativize(entry).toString().replace('\\', '/'));
        assertEquals("390a4471786704d20fb73bfe5529e32a77c71c7e81a4a90a13aae4cffbebacac",
                java.util.HexFormat.of().formatHex(
                        java.security.MessageDigest.getInstance("SHA-256").digest(bytes)));
        try (var siblings = Files.list(entry.getParent())) {
            assertFalse(siblings.anyMatch(path -> path.toString().endsWith(".tmp")));
        }

        clearWeakInterner();
        CpuGeneratedKernel loaded = new CpuGeneratedKernelArtifactStore(
                root.resolve(".")).loadOrGenerate(
                        specialization, new EmptyEmitter(LOWERING, emissions));
        assertNotSame(first, loaded);
        assertArrayEquals(first.classBytes(), loaded.classBytes());
        assertEquals(1, emissions.get());
        loaded.entryPoint().invokeExact(0L);
    }

    @Test void liveArtifactIsWeaklyInternedAndCallerHeldArtifactRemainsInvocable() throws Throwable {
        var emissions = new AtomicInteger();
        var firstStore = new CpuGeneratedKernelArtifactStore(temporaryDirectory);
        var secondStore = new CpuGeneratedKernelArtifactStore(temporaryDirectory.resolve("."));
        CpuGeneratedKernel first = firstStore.loadOrGenerate(
                specialization(1), new EmptyEmitter(LOWERING, emissions));
        assertSame(first, secondStore.loadOrGenerate(
                specialization(1), new EmptyEmitter(LOWERING, emissions)));
        clearAndEnqueueWeakInterner();
        first.entryPoint().invokeExact(0L);
        CpuGeneratedKernel reloaded = secondStore.loadOrGenerate(
                specialization(1), new EmptyEmitter(LOWERING, emissions));
        assertNotSame(first, reloaded);
        assertEquals(1, emissions.get());
    }

    @Test void timestampDoesNotInvalidateCompatibleEntry() throws Exception {
        var emissions = new AtomicInteger();
        CpuKernelSpecialization specialization = specialization(1);
        var store = new CpuGeneratedKernelArtifactStore(temporaryDirectory);
        store.loadOrGenerate(specialization, new EmptyEmitter(LOWERING, emissions));
        Path entry = onlyEntry(temporaryDirectory);
        Files.setLastModifiedTime(entry, FileTime.from(Instant.EPOCH));
        clearWeakInterner();
        store.loadOrGenerate(specialization, new EmptyEmitter(LOWERING, emissions));
        assertEquals(1, emissions.get());
    }

    @Test void laterJvmReusesStoredBytesAndConcurrentJvmWritersLeaveOneValidEnvelope()
            throws Exception {
        Path root = temporaryDirectory.resolve("processes");
        var emissions = new AtomicInteger();
        new CpuGeneratedKernelArtifactStore(root).loadOrGenerate(
                specialization(1), new EmptyEmitter(LOWERING, emissions));
        clearWeakInterner();
        assertEquals(0, runChild("read", root));

        Files.delete(onlyEntry(root));
        Process first = startChild("write", root);
        Process second = startChild("write", root);
        assertEquals(0, awaitChild(first));
        assertEquals(0, awaitChild(second));
        try (var paths = Files.walk(root)) {
            assertEquals(1, paths.filter(
                    path -> path.toString().endsWith(".cpuclass")).count());
        }
        clearWeakInterner();
        var rejectedEmission = new AtomicInteger();
        new CpuGeneratedKernelArtifactStore(root).loadOrGenerate(
                specialization(1), new EmptyEmitter(LOWERING, rejectedEmission));
        assertEquals(0, rejectedEmission.get());
    }

    @Test void validArtifactAppearingDuringEmissionIsPreferredWithoutReplacement() throws Exception {
        Path sourceRoot = temporaryDirectory.resolve("source");
        var sourceStore = new CpuGeneratedKernelArtifactStore(sourceRoot);
        sourceStore.loadOrGenerate(specialization(1), new EmptyEmitter(LOWERING, null));
        Path sourceEntry = onlyEntry(sourceRoot);
        byte[] envelope = Files.readAllBytes(sourceEntry);
        Path relative = sourceRoot.relativize(sourceEntry);
        clearWeakInterner();

        Path targetRoot = temporaryDirectory.resolve("target");
        Path targetEntry = targetRoot.resolve(relative);
        var control = new EmissionControl();
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<CpuGeneratedKernel> future = executor.submit(() ->
                    new CpuGeneratedKernelArtifactStore(targetRoot).loadOrGenerate(
                            specialization(1), control));
            assertTrue(control.entered.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
            Files.createDirectories(targetEntry.getParent());
            Files.write(targetEntry, envelope);
            FileTime appearedTime = FileTime.from(Instant.EPOCH.plusSeconds(17));
            Files.setLastModifiedTime(targetEntry, appearedTime);
            control.release.countDown();
            assertNotNull(future.get(TIMEOUT_SECONDS, TimeUnit.SECONDS));
            assertEquals(appearedTime, Files.getLastModifiedTime(targetEntry));
            assertEquals(1, control.emissions.get());
        } finally {
            control.release.countDown();
            executor.shutdownNow();
        }
    }

    @Test void metadataChecksumTruncationAndTrailingBytesRegenerateBeforeLoading() throws Exception {
        assertCorruptionRegenerates(bytes -> { bytes[0] ^= 1; return bytes; });
        assertCorruptionRegenerates(bytes -> {
            java.nio.ByteBuffer.wrap(bytes).putInt(8, 2); return bytes;
        });
        assertCorruptionRegenerates(bytes -> {
            java.nio.ByteBuffer.wrap(bytes).putInt(12, -1); return bytes;
        });
        assertCorruptionRegenerates(bytes -> { bytes[52] ^= 1; return bytes; });
        assertCorruptionRegenerates(bytes -> { bytes[20] ^= 1; return bytes; });
        assertCorruptionRegenerates(bytes -> java.util.Arrays.copyOf(bytes, bytes.length - 1));
        assertCorruptionRegenerates(bytes -> java.util.Arrays.copyOf(bytes, bytes.length + 1));
    }

    @Test void atomicMoveFailureHasNoFallbackAndCleansOnlyItsTemporaryFile() throws Exception {
        Path sourceRoot = temporaryDirectory.resolve("source-for-path");
        new CpuGeneratedKernelArtifactStore(sourceRoot).loadOrGenerate(
                specialization(1), new EmptyEmitter(LOWERING, null));
        Path relative = sourceRoot.relativize(onlyEntry(sourceRoot));
        clearWeakInterner();

        Path targetRoot = temporaryDirectory.resolve("atomic-failure");
        Path finalPath = targetRoot.resolve(relative);
        Files.createDirectories(finalPath);
        IllegalStateException failure = assertThrows(IllegalStateException.class,
                () -> new CpuGeneratedKernelArtifactStore(targetRoot).loadOrGenerate(
                        specialization(1), new EmptyEmitter(LOWERING, null)));
        assertTrue(failure.getMessage().startsWith(
                "failed to atomically publish generated-kernel artifact:"));
        assertNotNull(failure.getCause());
        assertTrue(Files.isDirectory(finalPath));
        try (var siblings = Files.list(finalPath.getParent())) {
            assertFalse(siblings.anyMatch(path -> path.getFileName().toString()
                    .startsWith(".cpuclass-")));
        }
    }

    @Test void unequalStructuralMetadataCannotAliasAtAnExistingPath() throws Exception {
        Path root = temporaryDirectory.resolve("collision");
        var emissions = new AtomicInteger();
        var store = new CpuGeneratedKernelArtifactStore(root);
        store.loadOrGenerate(specialization(1), new EmptyEmitter(LOWERING, emissions));
        Path first = onlyEntry(root);
        byte[] firstEnvelope = Files.readAllBytes(first);
        store.loadOrGenerate(specialization(2), new EmptyEmitter(LOWERING, emissions));
        Path second;
        try (var paths = Files.walk(root)) {
            second = paths.filter(path -> path.toString().endsWith(".cpuclass"))
                    .filter(path -> !path.equals(first)).findFirst().orElseThrow();
        }
        Files.write(second, firstEnvelope);
        clearWeakInterner();
        CpuGeneratedKernel repaired = store.loadOrGenerate(
                specialization(2), new EmptyEmitter(LOWERING, emissions));
        assertEquals(3, emissions.get());
        assertEquals(2, repaired.specialization().tileElementCount());
        assertFalse(java.util.Arrays.equals(firstEnvelope, Files.readAllBytes(second)));
    }

    @Test void equalConcurrentRequestsAcrossStoresShareOneAttempt() throws Exception {
        var control = new EmissionControl();
        var barrier = new CyclicBarrier(8);
        ExecutorService executor = Executors.newFixedThreadPool(8);
        try {
            List<Future<CpuGeneratedKernel>> futures = new ArrayList<>();
            for (int index = 0; index < 8; index++) {
                int storeIndex = index;
                futures.add(executor.submit(() -> {
                    barrier.await();
                    return new CpuGeneratedKernelArtifactStore(
                            temporaryDirectory.resolve(storeIndex % 2 == 0 ? "." : "x/.."))
                            .loadOrGenerate(specialization(1), control);
                }));
            }
            assertTrue(control.entered.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
            control.release.countDown();
            CpuGeneratedKernel expected = futures.getFirst().get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            for (Future<CpuGeneratedKernel> future : futures) {
                assertSame(expected, future.get(TIMEOUT_SECONDS, TimeUnit.SECONDS));
            }
            assertEquals(1, control.emissions.get());
        } finally {
            control.release.countDown();
            executor.shutdownNow();
        }
    }

    @Test void sharedFailureIsExactRetriableAndPreservesWaiterInterruption() throws Exception {
        Path unusableRoot = temporaryDirectory.resolve("root-file");
        Files.write(unusableRoot, new byte[] {1});
        var control = new EmissionControl();
        var store = new CpuGeneratedKernelArtifactStore(unusableRoot);
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<Throwable> owner = executor.submit(() -> captureFailure(
                    () -> store.loadOrGenerate(specialization(1), control)));
            assertTrue(control.entered.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
            CountDownLatch waiterStarted = new CountDownLatch(1);
            Future<WaiterOutcome> waiter = executor.submit(() -> {
                waiterStarted.countDown();
                Thread.currentThread().interrupt();
                Throwable failure = captureFailure(() -> new CpuGeneratedKernelArtifactStore(
                        unusableRoot).loadOrGenerate(specialization(1), control));
                return new WaiterOutcome(failure, Thread.currentThread().isInterrupted());
            });
            assertTrue(waiterStarted.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
            control.release.countDown();
            Throwable expected = owner.get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            WaiterOutcome observed = waiter.get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            assertSame(expected, observed.failure());
            assertTrue(observed.interrupted());
            assertInstanceOf(IllegalStateException.class, expected);
            assertNotNull(expected.getCause());
        } finally {
            control.release.countDown();
            executor.shutdownNow();
        }

        Files.delete(unusableRoot);
        var retryEmissions = new AtomicInteger();
        assertNotNull(store.loadOrGenerate(
                specialization(1), new EmptyEmitter(LOWERING, retryEmissions)));
        assertEquals(1, retryEmissions.get());
    }

    @Test void unrelatedKeysProgressWhileOneEmissionIsBlocked() throws Exception {
        var blocked = new EmissionControl();
        var store = new CpuGeneratedKernelArtifactStore(temporaryDirectory);
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<CpuGeneratedKernel> first = executor.submit(() -> store.loadOrGenerate(
                    specialization(1), blocked));
            assertTrue(blocked.entered.await(TIMEOUT_SECONDS, TimeUnit.SECONDS));
            var unrelatedEmissions = new AtomicInteger();
            Future<CpuGeneratedKernel> unrelated = executor.submit(() -> store.loadOrGenerate(
                    specialization(2), new EmptyEmitter(LOWERING, unrelatedEmissions)));
            assertNotNull(unrelated.get(TIMEOUT_SECONDS, TimeUnit.SECONDS));
            assertFalse(first.isDone());
            assertEquals(1, unrelatedEmissions.get());
            blocked.release.countDown();
            assertNotNull(first.get(TIMEOUT_SECONDS, TimeUnit.SECONDS));
        } finally {
            blocked.release.countDown();
            executor.shutdownNow();
        }
    }

    private void assertCorruptionRegenerates(ByteMutation mutation) throws Exception {
        long childCount;
        try (var children = Files.list(temporaryDirectory)) {
            childCount = children.count();
        }
        Path root = Files.createDirectory(temporaryDirectory.resolve(
                "corruption-" + childCount));
        var emissions = new AtomicInteger();
        var store = new CpuGeneratedKernelArtifactStore(root);
        CpuKernelSpecialization specialization = specialization(1);
        store.loadOrGenerate(specialization, new EmptyEmitter(LOWERING, emissions));
        Path entry = onlyEntry(root);
        byte[] valid = Files.readAllBytes(entry);
        byte[] changed = mutation.mutate(valid.clone());
        Files.write(entry, changed);
        clearWeakInterner();
        CpuGeneratedKernel repaired = store.loadOrGenerate(
                specialization, new EmptyEmitter(LOWERING, emissions));
        assertEquals(2, emissions.get());
        assertArrayEquals(valid, Files.readAllBytes(entry));
        assertArrayEquals(repaired.classBytes(),
                new CpuClassFileKernelGenerator().generateClassBytes(
                        specialization, new EmptyEmitter(LOWERING, null)));
    }

    private static CpuKernelSpecialization specialization(long tile) {
        return new CpuKernelSpecialization(CpuGeneratorSchema.CURRENT_VERSION, LOWERING,
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD, List.of(), List.of(), 0, null,
                ByteOrder.LITTLE_ENDIAN, 1, tile, CpuKernelSpecialization.Tail.NONE,
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuKernelSpecialization.CombineOrder.FIXED, ClassFile.JAVA_26_VERSION);
    }

    private static Path onlyEntry(Path root) throws IOException {
        try (var paths = Files.walk(root)) {
            return paths.filter(path -> path.toString().endsWith(".cpuclass"))
                    .findFirst().orElseThrow();
        }
    }

    @SuppressWarnings("unchecked")
    private static void clearWeakInterner() throws Exception {
        var field = CpuGeneratedKernelArtifactStore.class.getDeclaredField("INTERNED");
        field.setAccessible(true);
        ((Map<Object, Object>) field.get(null)).clear();
    }

    @SuppressWarnings("unchecked")
    private static void clearAndEnqueueWeakInterner() throws Exception {
        var field = CpuGeneratedKernelArtifactStore.class.getDeclaredField("INTERNED");
        field.setAccessible(true);
        Map<Object, Object> interned = (Map<Object, Object>) field.get(null);
        assertFalse(interned.isEmpty());
        for (Object value : List.copyOf(interned.values())) {
            Reference<?> reference = (Reference<?>) value;
            reference.clear();
            assertTrue(reference.enqueue());
        }
    }

    private static Throwable captureFailure(ThrowingRunnable action) {
        try {
            action.run();
            return fail("expected failure");
        } catch (Throwable failure) {
            return failure;
        }
    }

    private static Process startChild(String mode, Path root) throws IOException {
        Path java = Path.of(System.getProperty("java.home"), "bin", "java");
        return new ProcessBuilder(java.toString(), "--add-modules", "jdk.incubator.vector",
                "-cp", System.getProperty("java.class.path"),
                CpuGeneratedKernelArtifactStoreTest.class.getName(), mode, root.toString())
                .redirectErrorStream(true).start();
    }

    private static int runChild(String mode, Path root) throws Exception {
        return awaitChild(startChild(mode, root));
    }

    private static int awaitChild(Process process) throws Exception {
        if (!process.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
            process.destroyForcibly();
            fail("child JVM timed out");
        }
        byte[] output = process.getInputStream().readAllBytes();
        if (process.exitValue() != 0) {
            fail("child JVM failed: " + new String(output, java.nio.charset.StandardCharsets.UTF_8));
        }
        return process.exitValue();
    }

    public static void main(String[] arguments) throws Throwable {
        Path root = Path.of(arguments[1]);
        CpuFamilyKernelEmitter emitter = "read".equals(arguments[0])
                ? new EmptyEmitter(LOWERING, null) {
                    @Override public void emitScalar(CpuScalarEmitter scalar,
                            CpuCarrierEmitter carriers, CpuLoopEmitter loops,
                            CpuReductionEmitter reductions) {
                        throw new AssertionError("valid child-JVM hit must not emit");
                    }
                }
                : new EmptyEmitter(LOWERING, null);
        CpuGeneratedKernel artifact = new CpuGeneratedKernelArtifactStore(root)
                .loadOrGenerate(specialization(1), emitter);
        artifact.entryPoint().invokeExact(0L);
    }

    private static class EmptyEmitter implements CpuFamilyKernelEmitter {
        private final CpuLoweringFingerprint lowering;
        private final AtomicInteger emissions;

        private EmptyEmitter(CpuLoweringFingerprint lowering, AtomicInteger emissions) {
            this.lowering = lowering;
            this.emissions = emissions;
        }

        @Override public CpuLoweringFingerprint loweringFingerprint() { return lowering; }

        @Override public void emitScalar(CpuScalarEmitter scalar, CpuCarrierEmitter carriers,
                CpuLoopEmitter loops, CpuReductionEmitter reductions) {
            if (emissions != null) emissions.incrementAndGet();
            scalar.code().return_();
        }

        @Override public void emitVector(CpuVectorEmitter vector, CpuCarrierEmitter carriers,
                CpuLoopEmitter loops, CpuReductionEmitter reductions) {
            if (emissions != null) emissions.incrementAndGet();
            vector.code().return_();
        }
    }

    private static final class EmissionControl extends EmptyEmitter {
        private final AtomicInteger emissions = new AtomicInteger();
        private final CountDownLatch entered = new CountDownLatch(1);
        private final CountDownLatch release = new CountDownLatch(1);

        private EmissionControl() { super(LOWERING, null); }

        @Override public void emitScalar(CpuScalarEmitter scalar, CpuCarrierEmitter carriers,
                CpuLoopEmitter loops, CpuReductionEmitter reductions) {
            emissions.incrementAndGet();
            entered.countDown();
            try {
                if (!release.await(TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                    throw new AssertionError("timed out waiting to release emission");
                }
            } catch (InterruptedException failure) {
                Thread.currentThread().interrupt();
                throw new AssertionError(failure);
            }
            scalar.code().return_();
        }
    }

    @FunctionalInterface private interface ByteMutation { byte[] mutate(byte[] bytes); }
    @FunctionalInterface private interface ThrowingRunnable { void run() throws Throwable; }
    private record WaiterOutcome(Throwable failure, boolean interrupted) {}
}

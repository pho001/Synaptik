package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

final class CpuOpenBlasDiscoveryTest {
    @Test void requestModesPreserveExactValuesAndRejectInvalidRelationships() {
        String name = "  custom-openblas  ";
        Path path = Path.of("/custom/libopenblas.dylib");
        var disabled = CpuOpenBlasDiscoveryRequest.disabled();
        var automatic = CpuOpenBlasDiscoveryRequest.automatic();
        var exactName = CpuOpenBlasDiscoveryRequest.exactName(name);
        var exactPath = CpuOpenBlasDiscoveryRequest.exactAbsolutePath(path);

        assertAll(
                () -> assertEquals(CpuOpenBlasDiscoveryRequest.Mode.DISABLED, disabled.mode()),
                () -> assertEquals(CpuOpenBlasDiscoveryRequest.Mode.AUTOMATIC, automatic.mode()),
                () -> assertSame(name, exactName.exactName().orElseThrow()),
                () -> assertSame(path, exactPath.exactAbsolutePath().orElseThrow()),
                () -> assertThrows(NullPointerException.class,
                        () -> CpuOpenBlasDiscoveryRequest.exactName(null)),
                () -> assertThrows(NullPointerException.class,
                        () -> CpuOpenBlasDiscoveryRequest.exactAbsolutePath(null)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> CpuOpenBlasDiscoveryRequest.exactName(" \t")),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> CpuOpenBlasDiscoveryRequest.exactAbsolutePath(Path.of("relative"))),
                () -> assertThrows(NullPointerException.class,
                        () -> new CpuOpenBlasDiscoveryRequest(null, Optional.empty(),
                                Optional.empty())),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuOpenBlasDiscoveryRequest(
                                CpuOpenBlasDiscoveryRequest.Mode.AUTOMATIC, Optional.of("name"),
                                Optional.empty())),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuOpenBlasDiscoveryRequest(
                                CpuOpenBlasDiscoveryRequest.Mode.EXACT_NAME, Optional.empty(),
                                Optional.of(path))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuOpenBlasDiscoveryRequest(
                                CpuOpenBlasDiscoveryRequest.Mode.EXACT_ABSOLUTE_PATH,
                                Optional.of("name"), Optional.of(path))));
    }

    @Test void automaticCandidateTablesAndNormalizationAreExact() {
        assertAll(
                () -> assertEquals(List.of("name:openblas", "name:libopenblas.dylib",
                                "path:/opt/homebrew/opt/openblas/lib/libopenblas.dylib",
                                "path:/usr/local/opt/openblas/lib/libopenblas.dylib"),
                        candidates("  Darwin 23 ", " ARM64 ")),
                () -> assertEquals(List.of("name:openblas", "name:libopenblas.dylib",
                                "path:/usr/local/opt/openblas/lib/libopenblas.dylib",
                                "path:/opt/homebrew/opt/openblas/lib/libopenblas.dylib"),
                        candidates("Mac OS X", "x86_64")),
                () -> assertEquals(List.of("name:openblas", "name:libopenblas.so.0",
                                "name:libopenblas.so"),
                        candidates(" LINUX GNU ", "AARCH64")),
                () -> assertEquals(List.of("name:openblas", "name:libopenblas.dll",
                                "name:openblas.dll"),
                        candidates("Windows 11", "AMD64")),
                () -> assertEquals(List.of("name:openblas"), candidates("FreeBSD", "amd64")),
                () -> assertEquals(List.of("name:openblas"), candidates(null, null)),
                () -> assertEquals(CpuOpenBlasDiscoveryResult.OperatingSystemFamily.OTHER,
                        CpuOpenBlasDiscoveryResult.PlatformSnapshot.from(null, null)
                                .operatingSystemFamily()),
                () -> assertEquals("", CpuOpenBlasDiscoveryResult.PlatformSnapshot.from(null,
                        null).normalizedArchitecture()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuOpenBlasDiscoveryResult.PlatformSnapshot(
                                CpuOpenBlasDiscoveryResult.OperatingSystemFamily.LINUX,
                                "mac os x", "arm64")));
    }

    @Test void automaticSnapshotsPropertiesOnceAndStopsAtFirstSuccess() {
        var platform = new CountingPlatform(" Mac OS X ", " AARCH64 ");
        var loaded = new AtomicBoolean(true);
        var closes = new AtomicInteger();
        var attempted = new ArrayList<CpuOpenBlasDiscoveryResult.Selection>();
        CpuOpenBlasDiscovery.Loader loader = selection -> {
            attempted.add(selection);
            if (attempted.size() < 3) {
                throw new CpuOpenBlasDiscovery.LoadFailure("provider.Failure",
                        "failed " + attempted.size());
            }
            return owned(loaded, closes);
        };

        CpuOpenBlasDiscoverySession session = CpuOpenBlasDiscovery.discover(
                CpuOpenBlasDiscoveryRequest.automatic(), platform, loader);
        var result = session.result();
        assertAll(
                () -> assertEquals(1, platform.operatingSystemReads.get()),
                () -> assertEquals(1, platform.architectureReads.get()),
                () -> assertEquals(3, attempted.size()),
                () -> assertEquals(CpuOpenBlasDiscoveryResult.Status.LOADED, result.status()),
                () -> assertEquals(List.of(
                        CpuOpenBlasDiscoveryResult.AttemptStatus.LOAD_OR_BIND_FAILURE,
                        CpuOpenBlasDiscoveryResult.AttemptStatus.LOAD_OR_BIND_FAILURE,
                        CpuOpenBlasDiscoveryResult.AttemptStatus.LOADED),
                        result.attempts().stream().map(
                                CpuOpenBlasDiscoveryResult.Attempt::status).toList()),
                () -> assertEquals(attempted.getLast(), result.selected().orElseThrow()),
                () -> assertEquals("mac os x", result.platformSnapshot().orElseThrow()
                        .normalizedOperatingSystem()),
                () -> assertEquals("aarch64", result.platformSnapshot().orElseThrow()
                        .normalizedArchitecture()),
                () -> assertTrue(session.invocation().orElseThrow().isOpen()));
        session.close();
        assertAll(() -> assertFalse(loaded.get()), () -> assertEquals(1, closes.get()));
    }

    @Test void automaticExhaustionRetainsOrderedFailureDiagnosticsOnly() {
        CpuOpenBlasDiscoverySession session = CpuOpenBlasDiscovery.discover(
                CpuOpenBlasDiscoveryRequest.automatic(), new LinuxPlatform(),
                selection -> {
                    int position = linuxPosition(selection);
                    throw new CpuOpenBlasDiscovery.LoadFailure("provider.LoadFailure",
                            position == 1 ? null : "candidate " + position);
                });
        var result = session.result();
        assertAll(
                () -> assertEquals(CpuOpenBlasDiscoveryResult.Status.UNAVAILABLE,
                        result.status()),
                () -> assertEquals(3, result.attempts().size()),
                () -> assertTrue(result.selected().isEmpty()),
                () -> assertTrue(session.invocation().isEmpty()),
                () -> assertEquals("provider.LoadFailure",
                        result.attempts().get(1).failureType().orElseThrow()),
                () -> assertTrue(result.attempts().get(1).failureMessage().isEmpty()));
        session.close();
    }

    @Test void disabledAndExactModesNeverReadPlatformOrAppendCandidates() {
        var platform = new ThrowingPlatform();
        var loads = new AtomicInteger();
        CpuOpenBlasDiscovery.Loader unexpectedLoader = selection -> {
            loads.incrementAndGet();
            throw new AssertionError("disabled loader called");
        };
        try (var disabled = CpuOpenBlasDiscovery.discover(
                CpuOpenBlasDiscoveryRequest.disabled(), platform, unexpectedLoader)) {
            assertAll(
                    () -> assertEquals(CpuOpenBlasDiscoveryResult.Status.DISABLED,
                            disabled.result().status()),
                    () -> assertTrue(disabled.result().attempts().isEmpty()),
                    () -> assertTrue(disabled.invocation().isEmpty()),
                    () -> assertEquals(0, loads.get()));
        }

        var names = new ArrayList<CpuOpenBlasDiscoveryResult.Selection>();
        try (var exact = CpuOpenBlasDiscovery.discover(
                CpuOpenBlasDiscoveryRequest.exactName("exact-name"), platform, selection -> {
                    names.add(selection);
                    throw new CpuOpenBlasDiscovery.LoadFailure("provider.Failure", "missing");
                })) {
            assertAll(
                    () -> assertEquals(List.of("name:exact-name"), descriptions(names)),
                    () -> assertEquals(CpuOpenBlasDiscoveryResult.Status.UNAVAILABLE,
                            exact.result().status()),
                    () -> assertTrue(exact.result().platformSnapshot().isEmpty()));
        }

        Path exactPath = Path.of("/exact/not-probed/libopenblas.so");
        var paths = new ArrayList<CpuOpenBlasDiscoveryResult.Selection>();
        try (var exact = CpuOpenBlasDiscovery.discover(
                CpuOpenBlasDiscoveryRequest.exactAbsolutePath(exactPath), platform,
                selection -> {
                    paths.add(selection);
                    throw new CpuOpenBlasDiscovery.LoadFailure("provider.Failure", null);
                })) {
            assertAll(
                    () -> assertEquals(List.of("path:" + exactPath), descriptions(paths)),
                    () -> assertEquals(1, exact.result().attempts().size()),
                    () -> assertEquals(exactPath,
                            ((CpuOpenBlasDiscoveryResult.AbsoluteLibraryPath) paths.getFirst())
                                    .value()));
        }
    }

    @Test void resultDefensivelyCopiesAttemptsAndRejectsInconsistentMetadata() {
        var selection = new CpuOpenBlasDiscoveryResult.LibraryName("exact");
        var attempts = new ArrayList<>(List.of(CpuOpenBlasDiscoveryResult.Attempt.failed(0,
                selection, "provider.Failure", "missing")));
        var result = new CpuOpenBlasDiscoveryResult(
                CpuOpenBlasDiscoveryRequest.Mode.EXACT_NAME, Optional.empty(), attempts,
                Optional.empty(), CpuOpenBlasDiscoveryResult.Status.UNAVAILABLE);
        attempts.clear();
        assertAll(
                () -> assertEquals(1, result.attempts().size()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> result.attempts().clear()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuOpenBlasDiscoveryResult(
                                CpuOpenBlasDiscoveryRequest.Mode.DISABLED, Optional.empty(),
                                List.of(CpuOpenBlasDiscoveryResult.Attempt.loaded(0, selection)),
                                Optional.of(selection), CpuOpenBlasDiscoveryResult.Status.LOADED)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuOpenBlasDiscoveryResult(
                                CpuOpenBlasDiscoveryRequest.Mode.AUTOMATIC, Optional.empty(),
                                List.of(CpuOpenBlasDiscoveryResult.Attempt.failed(0, selection,
                                        "provider.Failure", null)), Optional.empty(),
                                CpuOpenBlasDiscoveryResult.Status.UNAVAILABLE)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuOpenBlasDiscoveryResult(
                                CpuOpenBlasDiscoveryRequest.Mode.EXACT_NAME, Optional.empty(),
                                List.of(CpuOpenBlasDiscoveryResult.Attempt.failed(1, selection,
                                        "provider.Failure", null)), Optional.empty(),
                                CpuOpenBlasDiscoveryResult.Status.UNAVAILABLE)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuOpenBlasDiscoveryResult(
                                CpuOpenBlasDiscoveryRequest.Mode.EXACT_NAME, Optional.empty(),
                                List.of(CpuOpenBlasDiscoveryResult.Attempt.loaded(0, selection)),
                                Optional.empty(), CpuOpenBlasDiscoveryResult.Status.LOADED)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuOpenBlasDiscoveryResult(
                                CpuOpenBlasDiscoveryRequest.Mode.AUTOMATIC,
                                Optional.of(CpuOpenBlasDiscoveryResult.PlatformSnapshot.from(
                                        "Linux", "amd64")),
                                List.of(CpuOpenBlasDiscoveryResult.Attempt.failed(0,
                                        new CpuOpenBlasDiscoveryResult.LibraryName("wrong"),
                                        "provider.Failure", null)), Optional.empty(),
                                CpuOpenBlasDiscoveryResult.Status.UNAVAILABLE)));
    }

    @Test void attemptsValidateDiagnosticRelationshipsAndRetainNoThrowable() {
        var selection = new CpuOpenBlasDiscoveryResult.LibraryName("exact");
        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> CpuOpenBlasDiscoveryResult.Attempt.failed(-1, selection,
                                "provider.Failure", null)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuOpenBlasDiscoveryResult.Attempt(0, selection,
                                CpuOpenBlasDiscoveryResult.AttemptStatus.LOADED,
                                Optional.of("provider.Failure"), Optional.empty())),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuOpenBlasDiscoveryResult.Attempt(0, selection,
                                CpuOpenBlasDiscoveryResult.AttemptStatus.LOAD_OR_BIND_FAILURE,
                                Optional.empty(), Optional.empty())),
                () -> assertTrue(java.util.Arrays.stream(
                                CpuOpenBlasDiscoveryResult.class.getDeclaredFields())
                        .noneMatch(field -> Throwable.class.isAssignableFrom(field.getType()))),
                () -> assertTrue(java.util.Arrays.stream(
                                CpuOpenBlasDiscoveryResult.Attempt.class.getDeclaredFields())
                        .noneMatch(field -> Throwable.class.isAssignableFrom(field.getType()))));
    }

    @Test void checkedLoaderFailureCopiesOnlyItsStrings() throws Exception {
        var failure = new CpuOpenBlasDiscovery.LoadFailure("provider.OpenBlasLoadException", null);
        assertAll(
                () -> assertEquals("provider.OpenBlasLoadException", failure.failureType()),
                () -> assertNull(failure.failureMessage()),
                () -> assertNull(failure.getCause()),
                () -> assertEquals(0, failure.getStackTrace().length),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuOpenBlasDiscovery.LoadFailure(" ", "message")));
    }

    @Test void unrelatedLoaderFailuresAndFatalErrorsPropagateUnchanged() {
        var programmingFailure = new IllegalStateException("programming defect");
        var fatal = new TestVirtualMachineError();
        var platformFailure = new IllegalStateException("property access denied");
        assertAll(
                () -> assertSame(programmingFailure, assertThrows(IllegalStateException.class,
                        () -> CpuOpenBlasDiscovery.discover(
                                CpuOpenBlasDiscoveryRequest.exactName("exact"),
                                new ThrowingPlatform(), selection -> {
                                    throw programmingFailure;
                                }))),
                () -> assertSame(fatal, assertThrows(TestVirtualMachineError.class,
                        () -> CpuOpenBlasDiscovery.discover(
                                CpuOpenBlasDiscoveryRequest.exactName("exact"),
                                new ThrowingPlatform(), selection -> {
                                    throw fatal;
                                }))),
                () -> assertThrows(NullPointerException.class,
                        () -> CpuOpenBlasDiscovery.discover(
                                CpuOpenBlasDiscoveryRequest.exactName("exact"),
                                new ThrowingPlatform(), selection -> null)),
                () -> assertSame(platformFailure, assertThrows(IllegalStateException.class,
                        () -> CpuOpenBlasDiscovery.discover(
                                CpuOpenBlasDiscoveryRequest.automatic(),
                                new CpuOpenBlasDiscovery.PlatformProperties() {
                                    @Override public String operatingSystem() {
                                        throw platformFailure;
                                    }

                                    @Override public String architecture() {
                                        throw new AssertionError("architecture read after failure");
                                    }
                                }, selection -> {
                                    throw new AssertionError("loader called after platform failure");
                                }))));
    }

    @Test void sessionOwnsOneCloseAndKeepsMetadataAndInvocationObservable() throws Exception {
        var open = new AtomicBoolean(true);
        var closes = new AtomicInteger();
        CpuOpenBlasDiscoverySession session = CpuOpenBlasDiscovery.discover(
                CpuOpenBlasDiscoveryRequest.exactName("exact"), new ThrowingPlatform(),
                selection -> owned(open, closes));
        CpuOpenBlasInvocation invocation = session.invocation().orElseThrow();
        var result = session.result();

        int count = 12;
        var ready = new CountDownLatch(count);
        var start = new CountDownLatch(1);
        var threads = new ArrayList<Thread>();
        for (int index = 0; index < count; index++) {
            Thread thread = new Thread(() -> {
                ready.countDown();
                try {
                    start.await();
                    session.close();
                } catch (InterruptedException failure) {
                    Thread.currentThread().interrupt();
                    throw new AssertionError(failure);
                }
            });
            threads.add(thread);
            thread.start();
        }
        ready.await();
        start.countDown();
        for (Thread thread : threads) thread.join();
        session.close();

        assertAll(
                () -> assertEquals(1, closes.get()),
                () -> assertFalse(invocation.isOpen()),
                () -> assertSame(result, session.result()),
                () -> assertSame(invocation, session.invocation().orElseThrow()),
                () -> assertEquals(CpuOpenBlasDiscoveryResult.Status.LOADED,
                        session.result().status()));
    }

    @Test void closeFailureClaimsOwnershipWithoutRetry() {
        var closes = new AtomicInteger();
        var closeFailure = new IllegalStateException("close failed");
        var selection = new CpuOpenBlasDiscoveryResult.LibraryName("exact");
        var result = new CpuOpenBlasDiscoveryResult(
                CpuOpenBlasDiscoveryRequest.Mode.EXACT_NAME, Optional.empty(),
                List.of(CpuOpenBlasDiscoveryResult.Attempt.loaded(0, selection)),
                Optional.of(selection), CpuOpenBlasDiscoveryResult.Status.LOADED);
        var session = new CpuOpenBlasDiscoverySession(result, Optional.of(
                new CpuOpenBlasDiscoverySession.OwnedResource(new FakeInvocation(
                        new AtomicBoolean(true)), () -> {
                            closes.incrementAndGet();
                            throw closeFailure;
                        })));
        assertSame(closeFailure, assertThrows(IllegalStateException.class, session::close));
        assertDoesNotThrow(session::close);
        assertEquals(1, closes.get());
    }

    @Test void sessionRejectsResourceStatusDisagreement() {
        var disabled = new CpuOpenBlasDiscoveryResult(
                CpuOpenBlasDiscoveryRequest.Mode.DISABLED, Optional.empty(), List.of(),
                Optional.empty(), CpuOpenBlasDiscoveryResult.Status.DISABLED);
        var open = new AtomicBoolean(true);
        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuOpenBlasDiscoverySession(disabled,
                                Optional.of(owned(open, new AtomicInteger())))),
                () -> assertThrows(NullPointerException.class,
                        () -> new CpuOpenBlasDiscoverySession(disabled, null)));
    }

    @Test void loadingDoesNotQualifyOrCompleteTheExistingRouteConfiguration() {
        var open = new AtomicBoolean(true);
        try (var session = CpuOpenBlasDiscovery.discover(
                CpuOpenBlasDiscoveryRequest.exactName("exact"), new ThrowingPlatform(),
                selection -> owned(open, new AtomicInteger()))) {
            assertAll(
                    () -> assertEquals(CpuOpenBlasDiscoveryResult.Status.LOADED,
                            session.result().status()),
                    () -> assertFalse(CpuPartitionAnalysisInputs.OpenBlasRouteConfig.DISABLED
                            .complete()),
                    () -> assertTrue(java.util.Arrays.stream(
                                    CpuOpenBlasDiscoveryResult.class.getDeclaredFields())
                            .noneMatch(field -> field.getType()
                                    == CpuPartitionAnalysisInputs.OpenBlasRouteConfig.class)));
        }
    }

    @Test void newTypesRemainInternalAndDiscoveryIsFieldFree() {
        assertAll(
                () -> assertFalse(Modifier.isPublic(
                        CpuOpenBlasDiscoveryRequest.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        CpuOpenBlasDiscoveryResult.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(CpuOpenBlasDiscovery.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        CpuOpenBlasDiscoverySession.class.getModifiers())),
                () -> assertTrue(Modifier.isPublic(
                        CpuOpenBlasQualification.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        CpuOpenBlasQualifier.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        CpuOpenBlasBinaryInspector.class.getModifiers())),
                () -> assertTrue(CpuOpenBlasDiscoveryRequest.class.isRecord()),
                () -> assertTrue(CpuOpenBlasDiscoveryResult.class.isRecord()),
                () -> assertEquals(0, CpuOpenBlasDiscovery.class.getDeclaredFields().length),
                () -> assertEquals(Set.of(AutoCloseable.class),
                        Set.of(CpuOpenBlasDiscoverySession.class.getInterfaces())));
    }

    @Test void productionSourcesContainOnlyBoundedColdMechanisms() throws Exception {
        Path root = Path.of("src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/"
                + "nativeblas/openblas");
        String discovery = Files.readString(root.resolve("CpuOpenBlasDiscovery.java"));
        String session = Files.readString(root.resolve("CpuOpenBlasDiscoverySession.java"));
        String combined = discovery + session
                + Files.readString(root.resolve("CpuOpenBlasDiscoveryRequest.java"))
                + Files.readString(root.resolve("CpuOpenBlasDiscoveryResult.java"));
        assertAll(
                () -> assertTrue(discovery.contains("OpenBlasLibrary.open(name.value())")),
                () -> assertTrue(discovery.contains("OpenBlasLibrary.open(path.value())")),
                () -> assertFalse(discovery.contains(".threadCount(")),
                () -> assertFalse(discovery.contains(".setThreadCount(")),
                () -> assertFalse(combined.contains("ServiceLoader")),
                () -> assertFalse(combined.contains("Files.walk")),
                () -> assertFalse(combined.contains("Files.list")),
                () -> assertFalse(combined.contains("ProcessBuilder")),
                () -> assertFalse(combined.contains("HttpClient")),
                () -> assertFalse(combined.contains("Runtime.getRuntime().exec")),
                () -> assertFalse(combined.contains("System.getenv")),
                () -> assertFalse(combined.contains("new File")),
                () -> assertFalse(session.contains("setThreadCount")));
    }

    @Test void checkpointRejectsInvalidArgumentsBeforeNativeAccess() {
        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> CpuOpenBlasNativeCheckpoint.main(new String[0])),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> CpuOpenBlasNativeCheckpoint.main(new String[] {"first", "second"})),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> CpuOpenBlasNativeCheckpoint.main(new String[] {"relative/path"})));
    }

    private static List<String> candidates(String operatingSystem, String architecture) {
        return descriptions(CpuOpenBlasDiscovery.automaticCandidates(
                CpuOpenBlasDiscoveryResult.PlatformSnapshot.from(operatingSystem, architecture)));
    }

    private static List<String> descriptions(
            List<CpuOpenBlasDiscoveryResult.Selection> selections) {
        return selections.stream().map(CpuOpenBlasDiscoveryTest::description).toList();
    }

    private static String description(CpuOpenBlasDiscoveryResult.Selection selection) {
        return switch (selection) {
            case CpuOpenBlasDiscoveryResult.LibraryName name -> "name:" + name.value();
            case CpuOpenBlasDiscoveryResult.AbsoluteLibraryPath path -> "path:" + path.value();
        };
    }

    private static int linuxPosition(CpuOpenBlasDiscoveryResult.Selection selection) {
        return switch (description(selection)) {
            case "name:openblas" -> 0;
            case "name:libopenblas.so.0" -> 1;
            case "name:libopenblas.so" -> 2;
            default -> throw new AssertionError("unexpected Linux candidate " + selection);
        };
    }

    private static CpuOpenBlasDiscoverySession.OwnedResource owned(AtomicBoolean open,
            AtomicInteger closes) {
        return new CpuOpenBlasDiscoverySession.OwnedResource(new FakeInvocation(open), () -> {
            closes.incrementAndGet();
            open.set(false);
        });
    }

    private static final class CountingPlatform implements CpuOpenBlasDiscovery.PlatformProperties {
        private final String operatingSystem;
        private final String architecture;
        private final AtomicInteger operatingSystemReads = new AtomicInteger();
        private final AtomicInteger architectureReads = new AtomicInteger();

        private CountingPlatform(String operatingSystem, String architecture) {
            this.operatingSystem = operatingSystem;
            this.architecture = architecture;
        }

        @Override public String operatingSystem() {
            operatingSystemReads.incrementAndGet();
            return operatingSystem;
        }

        @Override public String architecture() {
            architectureReads.incrementAndGet();
            return architecture;
        }
    }

    private static final class LinuxPlatform implements CpuOpenBlasDiscovery.PlatformProperties {
        @Override public String operatingSystem() { return "Linux"; }
        @Override public String architecture() { return "amd64"; }
    }

    private static final class ThrowingPlatform implements CpuOpenBlasDiscovery.PlatformProperties {
        @Override public String operatingSystem() { throw new AssertionError("platform read"); }
        @Override public String architecture() { throw new AssertionError("platform read"); }
    }

    private static final class FakeInvocation implements CpuOpenBlasInvocation {
        private final AtomicBoolean open;

        private FakeInvocation(AtomicBoolean open) {
            this.open = open;
        }

        @Override public boolean isOpen() { return open.get(); }

        @Override public int threadCount() {
            if (!open.get()) throw new IllegalStateException("closed");
            return 1;
        }

        @Override public void sgemm(int m, int n, int k, float alpha, MemorySegment a,
                MemorySegment b, float beta, MemorySegment c) {
            if (!open.get()) throw new IllegalStateException("closed");
        }

        @Override public void dgemm(int m, int n, int k, double alpha, MemorySegment a,
                MemorySegment b, double beta, MemorySegment c) {
            if (!open.get()) throw new IllegalStateException("closed");
        }
    }

    private static final class TestVirtualMachineError extends VirtualMachineError {
        private static final long serialVersionUID = 1L;
    }
}

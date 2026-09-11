package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuConcurrencyBudget;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class CpuOpenBlasQualifierTest {
    @TempDir Path directory;

    @Test void normalizesOnlyClosedTargets() {
        assertAll(() -> assertEquals(CpuOpenBlasQualification.Machine.AARCH64,
                        CpuOpenBlasQualifier.target(" macOS ", " ARM64 ", 64,
                                ByteOrder.LITTLE_ENDIAN).machine()),
                () -> assertEquals(CpuOpenBlasQualification.Machine.AARCH64,
                        CpuOpenBlasQualifier.target("Darwin", "aarch64", 64,
                                ByteOrder.LITTLE_ENDIAN).machine()),
                () -> assertEquals(CpuOpenBlasQualification.Machine.X86_64,
                        CpuOpenBlasQualifier.target("Linux", "amd64", 64,
                                ByteOrder.LITTLE_ENDIAN).machine()),
                () -> assertEquals(CpuOpenBlasQualification.OperatingSystem.WINDOWS,
                        CpuOpenBlasQualifier.target("Windows 11", "x86_64", 64,
                                ByteOrder.LITTLE_ENDIAN).operatingSystem()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> CpuOpenBlasQualifier.target(null, "x86_64", 64,
                                ByteOrder.LITTLE_ENDIAN)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> CpuOpenBlasQualifier.target("Linux", null, 64,
                                ByteOrder.LITTLE_ENDIAN)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> CpuOpenBlasQualifier.target("Linux", "x86_64", 32,
                                ByteOrder.LITTLE_ENDIAN)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> CpuOpenBlasQualifier.target("Linux", "x86_64", 64,
                                ByteOrder.BIG_ENDIAN)));
    }

    @Test void nameLoadedQualificationIsSessionOnlyAndExercisesAllEvidence() {
        FakeInvocation fake = new FakeInvocation();
        fake.threads = 7;
        CpuOpenBlasCoordinator coordinator = coordinator(fake, nameResult("openblas"));
        CpuOpenBlasQualification qualification = CpuOpenBlasQualifier.qualify(
                nameResult("openblas"), coordinator, linuxX86(),
                new CpuOpenBlasBinaryInspector(path -> fail("name load inspected a path")));
        assertAll(() -> assertEquals(CpuOpenBlasQualification.Scope.SESSION_ONLY,
                        qualification.scope()),
                () -> assertTrue(qualification.persistentIdentity().isEmpty()),
                () -> assertEquals(CpuOpenBlasQualification.REQUIRED_SYMBOLS,
                        qualification.requiredSymbols()),
                () -> assertEquals(CpuOpenBlasQualification.BlasIntAbi.C_INT_32,
                        qualification.blasIntAbi()),
                () -> assertEquals(10, fake.gemmCalls),
                () -> assertEquals(1, fake.threads));
        coordinator.validateQualification(qualification);
        coordinator.close();
        assertAll(() -> assertEquals(7, fake.threads), () -> assertFalse(fake.open.get()));
    }

    @Test void pathQualificationHasStableCompleteIdentityAndRejectsForeignSession()
            throws Exception {
        Path binary = directory.resolve("libopenblas.so");
        Files.write(binary, elfX86());
        FakeInvocation firstFake = new FakeInvocation();
        CpuOpenBlasDiscoveryResult firstResult = pathResult(binary);
        CpuOpenBlasCoordinator first = coordinator(firstFake, firstResult);
        CpuOpenBlasQualification one = CpuOpenBlasQualifier.qualify(firstResult, first, linuxX86(),
                new CpuOpenBlasBinaryInspector());
        FakeInvocation secondFake = new FakeInvocation();
        CpuOpenBlasDiscoveryResult secondResult = pathResult(binary);
        CpuOpenBlasCoordinator second = coordinator(secondFake, secondResult);
        CpuOpenBlasQualification two = CpuOpenBlasQualifier.qualify(secondResult, second, linuxX86(),
                new CpuOpenBlasBinaryInspector());
        assertAll(() -> assertEquals(CpuOpenBlasQualification.Scope.PERSISTENT_BINARY, one.scope()),
                () -> assertEquals(one.persistentIdentity(), two.persistentIdentity()),
                () -> assertEquals(Files.size(binary), one.binaryIdentity().orElseThrow().byteLength()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> second.validateQualification(one)));
        first.validateQualification(one);
        first.validateQualification(one);
        first.close(); second.close();
    }

    @Test void failedNumericalEvidenceIssuesNothingAndRestoresImmediately() {
        FakeInvocation fake = new FakeInvocation();
        fake.threads = 5;
        fake.corruptFinite = true;
        CpuOpenBlasCoordinator coordinator = coordinator(fake, nameResult("broken"));
        var failure = assertThrows(CpuOpenBlasQualification.QualificationException.class,
                () -> CpuOpenBlasQualifier.qualify(nameResult("broken"), coordinator, linuxX86(),
                        new CpuOpenBlasBinaryInspector()));
        assertAll(() -> assertNotNull(failure.getCause()), () -> assertEquals(5, fake.threads),
                () -> assertTrue(coordinator.isOpen()));
        coordinator.close();
    }

    @Test void persistentMembersAndDigestValidationAreExact() {
        var binary = new CpuOpenBlasQualification.BinaryIdentity(1, "SHA-256", "ab".repeat(32),
                64, CpuOpenBlasQualification.ExecutableFormat.ELF_64,
                CpuOpenBlasQualification.Machine.X86_64);
        var identity = new CpuOpenBlasQualification.PersistentIdentity(1, linuxX86(),
                CpuOpenBlasQualification.REQUIRED_SYMBOLS,
                CpuOpenBlasQualification.BlasIntAbi.C_INT_32,
                CpuOpenBlasQualification.NUMERICAL_CASE_VERSION, binary);
        assertAll(() -> assertEquals(List.copyOf(CpuOpenBlasQualification.REQUIRED_SYMBOLS),
                        identity.requiredSymbols()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuOpenBlasQualification.BinaryIdentity(1, "SHA-256",
                                "AB".repeat(32), 64,
                                CpuOpenBlasQualification.ExecutableFormat.ELF_64,
                                CpuOpenBlasQualification.Machine.X86_64)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuOpenBlasQualification.PersistentIdentity(2, linuxX86(),
                                CpuOpenBlasQualification.REQUIRED_SYMBOLS,
                                CpuOpenBlasQualification.BlasIntAbi.C_INT_32,
                                CpuOpenBlasQualification.NUMERICAL_CASE_VERSION, binary)));
    }

    private static CpuOpenBlasCoordinator coordinator(FakeInvocation fake,
            CpuOpenBlasDiscoveryResult result) {
        CpuOpenBlasDiscoverySession session = new CpuOpenBlasDiscoverySession(result,
                Optional.of(new CpuOpenBlasDiscoverySession.OwnedResource(fake,
                        () -> fake.open.set(false))));
        return session.transferToCoordinator(new CpuConcurrencyBudget(4));
    }

    private static CpuOpenBlasDiscoveryResult nameResult(String name) {
        var selection = new CpuOpenBlasDiscoveryResult.LibraryName(name);
        return new CpuOpenBlasDiscoveryResult(CpuOpenBlasDiscoveryRequest.Mode.EXACT_NAME,
                Optional.empty(), List.of(CpuOpenBlasDiscoveryResult.Attempt.loaded(0, selection)),
                Optional.of(selection), CpuOpenBlasDiscoveryResult.Status.LOADED);
    }

    private static CpuOpenBlasDiscoveryResult pathResult(Path path) {
        var selection = new CpuOpenBlasDiscoveryResult.AbsoluteLibraryPath(path);
        return new CpuOpenBlasDiscoveryResult(
                CpuOpenBlasDiscoveryRequest.Mode.EXACT_ABSOLUTE_PATH, Optional.empty(),
                List.of(CpuOpenBlasDiscoveryResult.Attempt.loaded(0, selection)),
                Optional.of(selection), CpuOpenBlasDiscoveryResult.Status.LOADED);
    }

    private static CpuOpenBlasQualification.TargetFingerprint linuxX86() {
        return CpuOpenBlasQualifier.target("Linux", "x86_64", 64, ByteOrder.LITTLE_ENDIAN);
    }

    private static byte[] elfX86() {
        byte[] b = new byte[64]; b[0] = 0x7f; b[1] = 'E'; b[2] = 'L'; b[3] = 'F';
        b[4] = 2; b[5] = 1; b[18] = 62; return b;
    }

    private static final class FakeInvocation implements CpuOpenBlasInvocation {
        final AtomicBoolean open = new AtomicBoolean(true);
        int threads = 1;
        int gemmCalls;
        boolean corruptFinite;
        @Override public boolean isOpen() { return open.get(); }
        @Override public int threadCount() { return threads; }
        @Override public void setThreadCount(int value) { threads = value; }
        @Override public void sgemm(int m, int n, int k, float alpha, MemorySegment a,
                MemorySegment b, float beta, MemorySegment c) {
            gemmCalls++;
            for (int row = 0; row < m; row++) for (int column = 0; column < n; column++) {
                float sum = 0;
                for (int p = 0; p < k; p++) sum += a.getAtIndex(ValueLayout.JAVA_FLOAT,
                        row * k + p) * b.getAtIndex(ValueLayout.JAVA_FLOAT, p * n + column);
                c.setAtIndex(ValueLayout.JAVA_FLOAT, row * n + column,
                        corruptFinite && m == 2 ? sum + 10 : sum);
            }
        }
        @Override public void dgemm(int m, int n, int k, double alpha, MemorySegment a,
                MemorySegment b, double beta, MemorySegment c) {
            gemmCalls++;
            for (int row = 0; row < m; row++) for (int column = 0; column < n; column++) {
                double sum = 0;
                for (int p = 0; p < k; p++) sum += a.getAtIndex(ValueLayout.JAVA_DOUBLE,
                        row * k + p) * b.getAtIndex(ValueLayout.JAVA_DOUBLE, p * n + column);
                c.setAtIndex(ValueLayout.JAVA_DOUBLE, row * n + column,
                        corruptFinite && m == 2 ? sum + 10 : sum);
            }
        }
    }
}

package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;

/**
 * Field-free cold issuer for bounded target, binary, ABI, symbol, and general matrix
 * multiplication (GEMM) evidence. A successful result is a compatibility credential for the
 * exact loaded session; it is not binary authentication or broad ABI, numerical, determinism,
 * or performance certification.
 */
final class CpuOpenBlasQualifier {
    private CpuOpenBlasQualifier() { }

    /**
     * Qualifies one loaded discovery result against one exact transferred coordinator and the
     * current supported host target.
     * @param discovery immutable loaded discovery metadata
     * @param coordinator exact live owner of the transferred discovery resource
     * @return immutable successful qualification; never {@code null}
     * @throws NullPointerException if an argument is {@code null}
     * @throws CpuOpenBlasQualification.QualificationException if an ordinary check fails
     */
    static CpuOpenBlasQualification qualify(CpuOpenBlasDiscoveryResult discovery,
            CpuOpenBlasCoordinator coordinator) {
        return qualify(discovery, coordinator,
                target(System.getProperty("os.name"), System.getProperty("os.arch"),
                        Math.toIntExact(ValueLayout.ADDRESS.byteSize() * Byte.SIZE),
                        ByteOrder.nativeOrder()),
                new CpuOpenBlasBinaryInspector());
    }

    /**
     * Qualifies with injected immutable target facts and binary inspector for deterministic tests.
     * @param discovery immutable loaded discovery metadata
     * @param coordinator exact live owner of the transferred discovery resource
     * @param target supported immutable target facts
     * @param inspector bounded inspector used only for an absolute-path selection
     * @return immutable successful qualification; never {@code null}
     * @throws NullPointerException if an argument is {@code null}
     * @throws CpuOpenBlasQualification.QualificationException if an ordinary check fails
     */
    static CpuOpenBlasQualification qualify(CpuOpenBlasDiscoveryResult discovery,
            CpuOpenBlasCoordinator coordinator,
            CpuOpenBlasQualification.TargetFingerprint target,
            CpuOpenBlasBinaryInspector inspector) {
        Objects.requireNonNull(discovery, "discovery");
        Objects.requireNonNull(coordinator, "coordinator");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(inspector, "inspector");
        try {
            if (discovery.status() != CpuOpenBlasDiscoveryResult.Status.LOADED
                    || discovery.selected().isEmpty()) {
                throw new IllegalStateException("OpenBLAS discovery did not load all symbols");
            }
            Optional<CpuOpenBlasQualification.BinaryIdentity> binary = Optional.empty();
            CpuOpenBlasQualification.Scope scope = CpuOpenBlasQualification.Scope.SESSION_ONLY;
            var selection = discovery.selected().orElseThrow();
            if (selection instanceof CpuOpenBlasDiscoveryResult.AbsoluteLibraryPath path) {
                binary = Optional.of(inspector.inspect(path.value(), target).identity());
                scope = CpuOpenBlasQualification.Scope.PERSISTENT_BINARY;
            } else if (!(selection instanceof CpuOpenBlasDiscoveryResult.LibraryName)) {
                throw new IllegalStateException("unsupported OpenBLAS discovery selection");
            }
            coordinator.qualify(target, () -> runNumericalCases(coordinator.invocation()));
            return new CpuOpenBlasQualification(scope, target, binary,
                    coordinator.sessionKey());
        } catch (CpuOpenBlasQualification.QualificationException failure) {
            throw failure;
        } catch (Exception failure) {
            throw new CpuOpenBlasQualification.QualificationException(
                    "OpenBLAS qualification failed", failure);
        }
    }

    /**
     * Normalizes injected host facts into the exact closed supported target vocabulary.
     * @param osName raw snapshotted operating-system property, possibly {@code null}
     * @param architecture raw snapshotted architecture property, possibly {@code null}
     * @param addressWidthBits native address width in bits
     * @param byteOrder native byte order
     * @return supported immutable target fingerprint
     * @throws NullPointerException if {@code byteOrder} is {@code null}
     * @throws IllegalArgumentException if any fact is absent or unsupported
     */
    static CpuOpenBlasQualification.TargetFingerprint target(String osName, String architecture,
            int addressWidthBits, ByteOrder byteOrder) {
        String os = normalize(osName);
        String arch = normalize(architecture);
        CpuOpenBlasQualification.OperatingSystem operatingSystem;
        if (os.startsWith("mac") || os.startsWith("darwin")) {
            operatingSystem = CpuOpenBlasQualification.OperatingSystem.MACOS;
        } else if (os.startsWith("linux")) {
            operatingSystem = CpuOpenBlasQualification.OperatingSystem.LINUX;
        } else if (os.startsWith("windows")) {
            operatingSystem = CpuOpenBlasQualification.OperatingSystem.WINDOWS;
        } else throw new IllegalArgumentException("unsupported OpenBLAS operating system");
        CpuOpenBlasQualification.Machine machine = switch (arch) {
            case "aarch64", "arm64" -> CpuOpenBlasQualification.Machine.AARCH64;
            case "amd64", "x86_64" -> CpuOpenBlasQualification.Machine.X86_64;
            default -> throw new IllegalArgumentException("unsupported OpenBLAS architecture");
        };
        return new CpuOpenBlasQualification.TargetFingerprint(1, operatingSystem, machine,
                addressWidthBits, byteOrder);
    }

    private static String normalize(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
    }

    private static void runNumericalCases(CpuOpenBlasInvocation invocation) {
        finiteFloat(invocation);
        finiteDouble(invocation);
        specialFloat(invocation, Float.NaN, Float.NaN);
        specialFloat(invocation, Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY);
        specialFloat(invocation, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY);
        specialFloat(invocation, 0.0f, 0.0f);
        specialDouble(invocation, Double.NaN, Double.NaN);
        specialDouble(invocation, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
        specialDouble(invocation, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);
        specialDouble(invocation, 0.0, 0.0);
    }

    private static void finiteFloat(CpuOpenBlasInvocation invocation) {
        float[] a = {1, -2, 3, 4, 5, -6};
        float[] b = {7, 8, -9, 10, 11, -12};
        double[] expected = {58, -48, -83, 154};
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment left = arena.allocate(6 * 4L, 4);
            MemorySegment right = arena.allocate(6 * 4L, 4);
            MemorySegment output = arena.allocate(4 * 4L, 4);
            for (int i = 0; i < a.length; i++) left.setAtIndex(ValueLayout.JAVA_FLOAT, i, a[i]);
            for (int i = 0; i < b.length; i++) right.setAtIndex(ValueLayout.JAVA_FLOAT, i, b[i]);
            for (int i = 0; i < 4; i++) output.setAtIndex(ValueLayout.JAVA_FLOAT, i, 12345.5f);
            invocation.sgemm(2, 2, 3, 1.0f, left, right, 0.0f, output);
            for (int i = 0; i < 4; i++) checkFiniteFloat(
                    output.getAtIndex(ValueLayout.JAVA_FLOAT, i), expected[i], 3,
                    sumAbs(a, b, i / 2, i % 2, 2, 3));
        }
    }

    private static void finiteDouble(CpuOpenBlasInvocation invocation) {
        double[] a = {1, -2, 3, 4, 5, -6};
        double[] b = {7, 8, -9, 10, 11, -12};
        double[] expected = {58, -48, -83, 154};
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment left = arena.allocate(6 * 8L, 8);
            MemorySegment right = arena.allocate(6 * 8L, 8);
            MemorySegment output = arena.allocate(4 * 8L, 8);
            for (int i = 0; i < a.length; i++) left.setAtIndex(ValueLayout.JAVA_DOUBLE, i, a[i]);
            for (int i = 0; i < b.length; i++) right.setAtIndex(ValueLayout.JAVA_DOUBLE, i, b[i]);
            for (int i = 0; i < 4; i++) output.setAtIndex(ValueLayout.JAVA_DOUBLE, i, 12345.5);
            invocation.dgemm(2, 2, 3, 1.0, left, right, 0.0, output);
            for (int i = 0; i < 4; i++) checkFiniteDouble(
                    output.getAtIndex(ValueLayout.JAVA_DOUBLE, i), expected[i], 3,
                    sumAbs(a, b, i / 2, i % 2, 2, 3));
        }
    }

    private static void specialFloat(CpuOpenBlasInvocation invocation, float exceptional,
            float expected) {
        float[] a;
        float[] b;
        int k;
        if (Float.isNaN(exceptional)) { a = new float[] {1, Float.NaN, 2}; b = new float[] {3, 4, 5}; k = 3; }
        else if (exceptional == 0.0f) { a = new float[] {0.0f, 0.0f}; b = new float[] {2, 3}; k = 2; }
        else { a = new float[] {exceptional, 2}; b = new float[] {1, 0}; k = 2; }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment left = arena.allocate(k * 4L, 4);
            MemorySegment right = arena.allocate(k * 4L, 4);
            MemorySegment output = arena.allocate(4, 4);
            for (int i = 0; i < k; i++) { left.setAtIndex(ValueLayout.JAVA_FLOAT, i, a[i]); right.setAtIndex(ValueLayout.JAVA_FLOAT, i, b[i]); }
            output.set(ValueLayout.JAVA_FLOAT, 0, 77f);
            invocation.sgemm(1, 1, k, 1f, left, right, 0f, output);
            float actual = output.get(ValueLayout.JAVA_FLOAT, 0);
            if (Float.isNaN(expected) ? !Float.isNaN(actual)
                    : exceptional == 0.0f ? Float.floatToRawIntBits(actual) != 0
                    : actual != expected) throw new IllegalStateException(
                            "FLOAT32 qualification result class disagrees");
        }
    }

    private static void specialDouble(CpuOpenBlasInvocation invocation, double exceptional,
            double expected) {
        double[] a;
        double[] b;
        int k;
        if (Double.isNaN(exceptional)) { a = new double[] {1, Double.NaN, 2}; b = new double[] {3, 4, 5}; k = 3; }
        else if (exceptional == 0.0) { a = new double[] {0.0, 0.0}; b = new double[] {2, 3}; k = 2; }
        else { a = new double[] {exceptional, 2}; b = new double[] {1, 0}; k = 2; }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment left = arena.allocate(k * 8L, 8);
            MemorySegment right = arena.allocate(k * 8L, 8);
            MemorySegment output = arena.allocate(8, 8);
            for (int i = 0; i < k; i++) { left.setAtIndex(ValueLayout.JAVA_DOUBLE, i, a[i]); right.setAtIndex(ValueLayout.JAVA_DOUBLE, i, b[i]); }
            output.set(ValueLayout.JAVA_DOUBLE, 0, 77d);
            invocation.dgemm(1, 1, k, 1d, left, right, 0d, output);
            double actual = output.get(ValueLayout.JAVA_DOUBLE, 0);
            if (Double.isNaN(expected) ? !Double.isNaN(actual)
                    : exceptional == 0.0 ? Double.doubleToRawLongBits(actual) != 0L
                    : actual != expected) throw new IllegalStateException(
                            "FLOAT64 qualification result class disagrees");
        }
    }

    private static double sumAbs(float[] a, float[] b, int row, int column, int n, int k) {
        double sum = 0;
        for (int p = 0; p < k; p++) sum += Math.abs((double) a[row * k + p] * b[p * n + column]);
        return sum;
    }

    private static double sumAbs(double[] a, double[] b, int row, int column, int n, int k) {
        double sum = 0;
        for (int p = 0; p < k; p++) sum += Math.abs(a[row * k + p] * b[p * n + column]);
        return sum;
    }

    private static void checkFiniteFloat(float actual, double expected, int k, double sumAbs) {
        float rounded = (float) expected;
        double u = Math.scalb(1.0, -24);
        double gamma = 2.0 * k * u / (1.0 - 2.0 * k * u);
        double tolerance = Math.max(gamma * sumAbs, 4.0 * Math.ulp(rounded));
        if (!Float.isFinite(actual) || Math.abs(actual - expected) > tolerance) {
            throw new IllegalStateException("FLOAT32 finite qualification result disagrees");
        }
    }

    private static void checkFiniteDouble(double actual, double expected, int k, double sumAbs) {
        double u = Math.scalb(1.0, -53);
        double gamma = 2.0 * k * u / (1.0 - 2.0 * k * u);
        double tolerance = Math.max(gamma * sumAbs, 4.0 * Math.ulp(expected));
        if (!Double.isFinite(actual) || Math.abs(actual - expected) > tolerance) {
            throw new IllegalStateException("FLOAT64 finite qualification result disagrees");
        }
    }
}

package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.Objects;

/**
 * Bounded cold complete-file digest and closed 64-bit executable-header inspector. Inspection
 * reads a resolved regular file, validates ordinary before/after stability facts, and parses only
 * the supported Mach-O, ELF, or PE64 header. The resulting identity is compatibility evidence;
 * pathname inspection and SHA-256 do not authenticate an already loaded library or protect
 * against adversarial replacement.
 */
final class CpuOpenBlasBinaryInspector {
    /** Largest binary accepted for complete streaming inspection: one gibibyte. */
    static final long MAXIMUM_BINARY_BYTES = 1L << 30;
    private static final int HEADER_LIMIT = 65_536;
    private static final int BUFFER_SIZE = 64 * 1024;

    /** Post-read hook used to make file-stability changes deterministic in tests. */
    @FunctionalInterface
    interface InspectionHook {
        /**
         * Runs after the complete read and before final file-attribute validation.
         *
         * @param realPath resolved path that was inspected
         * @throws IOException if the injected inspection step fails
         */
        void afterRead(Path realPath) throws IOException;
    }

    /**
     * Successful identity plus diagnostic resolved path.
     *
     * @param identity stable complete-file compatibility identity
     * @param realPath resolved diagnostic path, excluded from identity equality
     */
    record Inspection(CpuOpenBlasQualification.BinaryIdentity identity, Path realPath) { }

    private final InspectionHook hook;

    /** Creates a production inspector with no injected post-read action. */
    CpuOpenBlasBinaryInspector() { this(path -> { }); }

    /**
     * Creates an inspector with a deterministic post-read test seam.
     *
     * @param hook non-null action invoked before final attribute validation
     * @throws NullPointerException if {@code hook} is {@code null}
     */
    CpuOpenBlasBinaryInspector(InspectionHook hook) {
        this.hook = Objects.requireNonNull(hook, "hook");
    }

    /**
     * Streams and identifies one exact supported absolute-path selection.
     *
     * @param selectedPath selected path to resolve and inspect; never written
     * @param target supported target that the executable header must match
     * @return immutable identity and diagnostic resolved path; never {@code null}
     * @throws NullPointerException if an argument is {@code null}
     * @throws IOException if path resolution, regular-file/size/stability validation, reading,
     *     or executable-header/target validation fails
     * @throws ArithmeticException if a changing stream exceeds numeric counting bounds
     */
    Inspection inspect(Path selectedPath,
            CpuOpenBlasQualification.TargetFingerprint target) throws IOException {
        Objects.requireNonNull(selectedPath, "selectedPath");
        Objects.requireNonNull(target, "target");
        Path real = selectedPath.toRealPath();
        BasicFileAttributes before = attributes(real);
        long expected = before.size();
        if (!before.isRegularFile() || expected <= 0 || expected > MAXIMUM_BINARY_BYTES) {
            throw new IOException("OpenBLAS binary must be a nonempty regular file at most 1 GiB: "
                    + real);
        }
        MessageDigest digest = sha256();
        byte[] header = new byte[(int) Math.min(expected, HEADER_LIMIT)];
        byte[] buffer = new byte[BUFFER_SIZE];
        long count = 0;
        int headerCount = 0;
        try (InputStream input = Files.newInputStream(real)) {
            for (int read; (read = input.read(buffer)) >= 0;) {
                if (read == 0) continue;
                count = Math.addExact(count, read);
                if (count > MAXIMUM_BINARY_BYTES) throw new IOException(
                        "OpenBLAS binary changed beyond the 1-GiB ceiling during inspection: "
                                + real);
                digest.update(buffer, 0, read);
                int copied = Math.min(read, header.length - headerCount);
                if (copied > 0) {
                    System.arraycopy(buffer, 0, header, headerCount, copied);
                    headerCount += copied;
                }
            }
        }
        hook.afterRead(real);
        BasicFileAttributes after = attributes(real);
        if (count != expected || !stable(before, after)) throw new IOException(
                "OpenBLAS binary changed during inspection: " + real);
        ParsedHeader parsed = parse(Arrays.copyOf(header, headerCount), target);
        var identity = new CpuOpenBlasQualification.BinaryIdentity(1, "SHA-256",
                HexFormat.of().formatHex(digest.digest()), count, parsed.format(),
                parsed.machine());
        return new Inspection(identity, real);
    }

    private static BasicFileAttributes attributes(Path real) throws IOException {
        return Files.readAttributes(real, BasicFileAttributes.class,
                LinkOption.NOFOLLOW_LINKS);
    }

    private static boolean stable(BasicFileAttributes before, BasicFileAttributes after) {
        Object beforeKey = before.fileKey();
        Object afterKey = after.fileKey();
        return after.isRegularFile() && before.size() == after.size()
                && before.lastModifiedTime().equals(after.lastModifiedTime())
                && (beforeKey == null || afterKey == null || beforeKey.equals(afterKey));
    }

    private static MessageDigest sha256() {
        try { return MessageDigest.getInstance("SHA-256"); }
        catch (NoSuchAlgorithmException impossible) { throw new AssertionError(impossible); }
    }

    private record ParsedHeader(CpuOpenBlasQualification.ExecutableFormat format,
            CpuOpenBlasQualification.Machine machine) { }

    private static ParsedHeader parse(byte[] bytes,
            CpuOpenBlasQualification.TargetFingerprint target) throws IOException {
        ParsedHeader parsed = switch (target.operatingSystem()) {
            case MACOS -> parseMachO(bytes);
            case LINUX -> parseElf(bytes);
            case WINDOWS -> parsePe(bytes);
        };
        if (parsed.machine() != target.machine()) throw new IOException(
                "OpenBLAS binary machine disagrees with host target");
        return parsed;
    }

    private static ParsedHeader parseMachO(byte[] b) throws IOException {
        require(b, 0, 8);
        boolean little;
        if (u32be(b, 0) == 0xcffaedfeL) little = true;
        else if (u32be(b, 0) == 0xfeedfacfL) little = false;
        else throw new IOException("OpenBLAS binary is not a supported thin 64-bit Mach-O");
        long cpu = little ? u32le(b, 4) : u32be(b, 4);
        CpuOpenBlasQualification.Machine machine = switch ((int) cpu) {
            case 0x0100000c -> CpuOpenBlasQualification.Machine.AARCH64;
            case 0x01000007 -> CpuOpenBlasQualification.Machine.X86_64;
            default -> throw new IOException("unsupported Mach-O machine");
        };
        return new ParsedHeader(CpuOpenBlasQualification.ExecutableFormat.MACH_O_64, machine);
    }

    private static ParsedHeader parseElf(byte[] b) throws IOException {
        require(b, 0, 20);
        if (u8(b, 0) != 0x7f || u8(b, 1) != 'E' || u8(b, 2) != 'L'
                || u8(b, 3) != 'F' || u8(b, 4) != 2 || u8(b, 5) != 1) {
            throw new IOException("OpenBLAS binary is not native-little-endian ELF64");
        }
        int machineCode = u16le(b, 18);
        CpuOpenBlasQualification.Machine machine = switch (machineCode) {
            case 183 -> CpuOpenBlasQualification.Machine.AARCH64;
            case 62 -> CpuOpenBlasQualification.Machine.X86_64;
            default -> throw new IOException("unsupported ELF machine");
        };
        return new ParsedHeader(CpuOpenBlasQualification.ExecutableFormat.ELF_64, machine);
    }

    private static ParsedHeader parsePe(byte[] b) throws IOException {
        require(b, 0, 64);
        if (u8(b, 0) != 'M' || u8(b, 1) != 'Z') throw new IOException(
                "OpenBLAS binary has no DOS header");
        long offsetLong = u32le(b, 0x3c);
        if (offsetLong > Integer.MAX_VALUE) throw new IOException("oversized PE header offset");
        int offset = (int) offsetLong;
        require(b, offset, 26);
        if (u8(b, offset) != 'P' || u8(b, offset + 1) != 'E'
                || u8(b, offset + 2) != 0 || u8(b, offset + 3) != 0
                || u16le(b, offset + 20) < 2
                || u16le(b, offset + 24) != 0x20b) {
            throw new IOException("OpenBLAS binary is not PE32+");
        }
        CpuOpenBlasQualification.Machine machine = switch (u16le(b, offset + 4)) {
            case 0xaa64 -> CpuOpenBlasQualification.Machine.AARCH64;
            case 0x8664 -> CpuOpenBlasQualification.Machine.X86_64;
            default -> throw new IOException("unsupported PE machine");
        };
        return new ParsedHeader(CpuOpenBlasQualification.ExecutableFormat.PE_32_PLUS, machine);
    }

    private static void require(byte[] bytes, int offset, int length) throws IOException {
        if (offset < 0 || length < 0 || offset > bytes.length - length) {
            throw new IOException("truncated or oversized executable header");
        }
    }

    private static int u8(byte[] b, int offset) { return b[offset] & 0xff; }
    private static int u16le(byte[] b, int offset) {
        return u8(b, offset) | u8(b, offset + 1) << 8;
    }
    private static long u32le(byte[] b, int offset) {
        return Integer.toUnsignedLong(u8(b, offset) | u8(b, offset + 1) << 8
                | u8(b, offset + 2) << 16 | u8(b, offset + 3) << 24);
    }
    private static long u32be(byte[] b, int offset) {
        return Integer.toUnsignedLong(u8(b, offset) << 24 | u8(b, offset + 1) << 16
                | u8(b, offset + 2) << 8 | u8(b, offset + 3));
    }
}

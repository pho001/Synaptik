package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class CpuOpenBlasBinaryInspectorTest {
    @TempDir Path directory;

    @Test void parsesEveryClosedTargetHeaderPairAndDigestsCompleteBytes() throws Exception {
        record Case(CpuOpenBlasQualification.OperatingSystem os,
                CpuOpenBlasQualification.Machine machine, byte[] bytes,
                CpuOpenBlasQualification.ExecutableFormat format) { }
        var cases = java.util.List.of(
                new Case(CpuOpenBlasQualification.OperatingSystem.MACOS,
                        CpuOpenBlasQualification.Machine.AARCH64, mach(0x0100000c),
                        CpuOpenBlasQualification.ExecutableFormat.MACH_O_64),
                new Case(CpuOpenBlasQualification.OperatingSystem.MACOS,
                        CpuOpenBlasQualification.Machine.X86_64, mach(0x01000007),
                        CpuOpenBlasQualification.ExecutableFormat.MACH_O_64),
                new Case(CpuOpenBlasQualification.OperatingSystem.LINUX,
                        CpuOpenBlasQualification.Machine.AARCH64, elf(183),
                        CpuOpenBlasQualification.ExecutableFormat.ELF_64),
                new Case(CpuOpenBlasQualification.OperatingSystem.LINUX,
                        CpuOpenBlasQualification.Machine.X86_64, elf(62),
                        CpuOpenBlasQualification.ExecutableFormat.ELF_64),
                new Case(CpuOpenBlasQualification.OperatingSystem.WINDOWS,
                        CpuOpenBlasQualification.Machine.AARCH64, pe(0xaa64),
                        CpuOpenBlasQualification.ExecutableFormat.PE_32_PLUS),
                new Case(CpuOpenBlasQualification.OperatingSystem.WINDOWS,
                        CpuOpenBlasQualification.Machine.X86_64, pe(0x8664),
                        CpuOpenBlasQualification.ExecutableFormat.PE_32_PLUS));
        int index = 0;
        for (Case value : cases) {
            byte[] complete = java.util.Arrays.copyOf(value.bytes(), value.bytes().length + 17);
            complete[complete.length - 1] = (byte) ++index;
            Path path = directory.resolve("binary-" + index);
            Files.write(path, complete);
            var inspected = new CpuOpenBlasBinaryInspector().inspect(path,
                    target(value.os(), value.machine()));
            assertAll(() -> assertEquals(path.toRealPath(), inspected.realPath()),
                    () -> assertEquals(value.format(), inspected.identity().executableFormat()),
                    () -> assertEquals(value.machine(), inspected.identity().machine()),
                    () -> assertEquals(complete.length, inspected.identity().byteLength()),
                    () -> assertEquals(HexFormat.of().formatHex(
                                    MessageDigest.getInstance("SHA-256").digest(complete)),
                            inspected.identity().sha256()));
        }
    }

    @Test void rejectsMismatchTruncationWrongClassEndianFatMachAndPeOffset() throws Exception {
        Path path = directory.resolve("bad");
        Files.write(path, elf(62));
        assertThrows(java.io.IOException.class, () -> new CpuOpenBlasBinaryInspector().inspect(path,
                target(CpuOpenBlasQualification.OperatingSystem.LINUX,
                        CpuOpenBlasQualification.Machine.AARCH64)));
        for (byte[] bytes : java.util.List.of(new byte[] {0x7f, 'E', 'L', 'F'},
                new byte[] {(byte) 0xca, (byte) 0xfe, (byte) 0xba, (byte) 0xbe, 0, 0, 0, 0},
                elf32(), elfBigEndian(), peWithOffset(70_000))) {
            Files.write(path, bytes);
            assertThrows(java.io.IOException.class, () -> new CpuOpenBlasBinaryInspector().inspect(
                    path, targetFor(bytes)));
        }
    }

    @Test void rejectsEmptyDirectoryAndMutationDuringInspection() throws Exception {
        assertThrows(java.io.IOException.class, () -> new CpuOpenBlasBinaryInspector().inspect(
                directory, target(CpuOpenBlasQualification.OperatingSystem.LINUX,
                        CpuOpenBlasQualification.Machine.X86_64)));
        Path empty = directory.resolve("empty");
        Files.write(empty, new byte[0]);
        assertThrows(java.io.IOException.class, () -> new CpuOpenBlasBinaryInspector().inspect(
                empty, target(CpuOpenBlasQualification.OperatingSystem.LINUX,
                        CpuOpenBlasQualification.Machine.X86_64)));
        Path mutable = directory.resolve("mutable");
        Files.write(mutable, elf(62));
        var inspector = new CpuOpenBlasBinaryInspector(real -> Files.write(real,
                java.util.Arrays.copyOf(elf(62), 65)));
        assertThrows(java.io.IOException.class, () -> inspector.inspect(mutable,
                target(CpuOpenBlasQualification.OperatingSystem.LINUX,
                        CpuOpenBlasQualification.Machine.X86_64)));
    }

    private static CpuOpenBlasQualification.TargetFingerprint target(
            CpuOpenBlasQualification.OperatingSystem os,
            CpuOpenBlasQualification.Machine machine) {
        return new CpuOpenBlasQualification.TargetFingerprint(1, os, machine, 64,
                ByteOrder.LITTLE_ENDIAN);
    }

    private static CpuOpenBlasQualification.TargetFingerprint targetFor(byte[] bytes) {
        return target(bytes.length >= 2 && bytes[0] == 'M'
                        ? CpuOpenBlasQualification.OperatingSystem.WINDOWS
                        : bytes.length >= 4 && bytes[0] == 0x7f
                                ? CpuOpenBlasQualification.OperatingSystem.LINUX
                                : CpuOpenBlasQualification.OperatingSystem.MACOS,
                CpuOpenBlasQualification.Machine.X86_64);
    }

    private static byte[] mach(int machine) {
        byte[] b = new byte[32];
        put32le(b, 0, 0xfeedfacf); put32le(b, 4, machine);
        return b;
    }
    private static byte[] elf(int machine) {
        byte[] b = new byte[64];
        b[0] = 0x7f; b[1] = 'E'; b[2] = 'L'; b[3] = 'F'; b[4] = 2; b[5] = 1;
        put16le(b, 18, machine); return b;
    }
    private static byte[] elf32() { byte[] b = elf(62); b[4] = 1; return b; }
    private static byte[] elfBigEndian() { byte[] b = elf(62); b[5] = 2; return b; }
    private static byte[] pe(int machine) {
        byte[] b = new byte[160]; b[0] = 'M'; b[1] = 'Z'; put32le(b, 0x3c, 96);
        b[96] = 'P'; b[97] = 'E'; put16le(b, 100, machine); put16le(b, 116, 2);
        put16le(b, 120, 0x20b);
        return b;
    }
    private static byte[] peWithOffset(int offset) {
        byte[] b = new byte[64]; b[0] = 'M'; b[1] = 'Z'; put32le(b, 0x3c, offset); return b;
    }
    private static void put16le(byte[] b, int offset, int value) {
        b[offset] = (byte) value; b[offset + 1] = (byte) (value >>> 8);
    }
    private static void put32le(byte[] b, int offset, int value) {
        put16le(b, offset, value); put16le(b, offset + 2, value >>> 16);
    }
}

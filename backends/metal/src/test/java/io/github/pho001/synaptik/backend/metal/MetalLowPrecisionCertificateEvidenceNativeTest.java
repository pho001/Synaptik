package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.trace.certificate.LowPrecisionCertificateKey;
import io.github.pho001.synaptik.trace.payload.TraceNumericalProfile;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Real Apple-GPU generator for the active schema-one raw-preserving certificate corpus. */
class MetalLowPrecisionCertificateEvidenceNativeTest {
    private static final int EXHAUSTIVE_WORDS = 1 << 16;
    private static final int REPEATS = 3;

    @Test
    void generatesEnvironmentBoundRawPreservingCertificates() throws IOException {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        String beforeDigest = MetalCertificationEnvironment.sha256(library);
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        try {
            context = api.createContext();
            String nativeRecord = api.certificationEnvironment(context);
            String afterDigest = MetalCertificationEnvironment.sha256(library);
            MetalCertificationEnvironment environment = MetalCertificationEnvironment.parse(
                    nativeRecord, beforeDigest, afterDigest);
            var observations = new ArrayList<Observation>();
            var certificates = new ArrayList<CertificateCase>();

            byte[] exhaustive = exhaustiveWords();
            for (NumericalProfile profile : NumericalProfile.values()) {
                for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
                    for (EvidenceCase evidenceCase : baseCases(type, exhaustive)) {
                        Observation observation = execute(
                                api, context, profile, evidenceCase);
                        observations.add(observation);
                        certificates.add(new CertificateCase(
                                profile, type, evidenceCase, observation.programDigest()));
                    }
                }
            }
            for (NumericalProfile profile : NumericalProfile.values()) {
                for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
                    for (EvidenceCase evidenceCase : expandedCases(type, exhaustive)) {
                        Observation observation = execute(
                                api, context, profile, evidenceCase);
                        observations.add(observation);
                        certificates.add(new CertificateCase(
                                profile, type, evidenceCase, observation.programDigest()));
                    }
                }
            }
            byte[] odd = sequentialWords(257);
            for (NumericalProfile profile : NumericalProfile.values()) {
                for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
                    for (EvidenceCase evidenceCase : oddCases(type, odd)) {
                        observations.add(execute(api, context, profile, evidenceCase));
                    }
                }
            }
            assertEquals(48, observations.size());
            assertEquals(24, certificates.size());
            writeEvidence(environment, observations, certificates);
        } finally {
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static Observation execute(
            MetalNativeApi api,
            MetalNativeApi.Handle context,
            NumericalProfile profile,
            EvidenceCase evidenceCase) {
        byte[] expected = evidenceCase.expected();
        MetalNativeApi.Handle executable = null;
        var inputs = new ArrayList<MetalNativeApi.Handle>();
        MetalNativeApi.Handle output = null;
        try {
            executable = api.createMpsGraphExecutable(
                    context,
                    profile,
                    evidenceCase.values(),
                    evidenceCase.program(),
                    evidenceCase.feeds(),
                    new int[] {evidenceCase.target()},
                    MetalPreparedRoute.MPSGRAPH);
            for (byte[] feed : evidenceCase.feedBytes()) {
                MetalNativeApi.Handle buffer = api.createBuffer(context, feed.length);
                inputs.add(buffer);
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment source = arena.allocate(feed.length, Short.BYTES);
                    source.copyFrom(MemorySegment.ofArray(feed));
                    api.upload(buffer, 0L, source, feed.length);
                }
            }
            output = api.createBuffer(
                    context, evidenceCase.values().get(evidenceCase.target()).byteCount());
            byte[] previous = null;
            for (int repetition = 0; repetition < REPEATS; repetition++) {
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment poison = arena.allocate(expected.length, Short.BYTES);
                    poison.fill((byte) (0x51 + repetition));
                    api.upload(output, 0L, poison, expected.length);
                    MemorySegment addresses = arena.allocate(ADDRESS, inputs.size());
                    for (int index = 0; index < inputs.size(); index++) {
                        addresses.setAtIndex(ADDRESS, index, inputs.get(index).carrier());
                    }
                    MemorySegment outputs = arena.allocate(ADDRESS);
                    outputs.set(ADDRESS, 0L, output.carrier());
                    try {
                        api.runExecutable(
                                executable, inputs.size(), addresses, 1, outputs);
                    } catch (RuntimeException failure) {
                        throw new IllegalStateException(
                                evidenceCase.name() + " execution failed", failure);
                    }
                    MemorySegment actual = arena.allocate(expected.length, Short.BYTES);
                    api.download(output, 0L, actual, expected.length);
                    byte[] bytes = actual.toArray(JAVA_BYTE);
                    assertArrayEquals(expected, bytes,
                            evidenceCase.name() + " exact raw output repetition " + repetition);
                    if (previous != null) {
                        assertArrayEquals(previous, bytes,
                                evidenceCase.name() + " deterministic repeat");
                    }
                    previous = bytes;
                }
            }
            byte[] image = evidenceCase.program().encodedProgramImage(
                    profile,
                    evidenceCase.values(),
                    evidenceCase.feeds(),
                    new int[] {evidenceCase.target()},
                    MetalPreparedRoute.MPSGRAPH);
            return new Observation(
                    evidenceCase.name(), profile, evidenceCase.type(),
                    expected.length / Short.BYTES, sha256(image), "PASS", REPEATS);
        } finally {
            if (output != null) api.releaseBuffer(output);
            for (int index = inputs.size(); index-- > 0;) api.releaseBuffer(inputs.get(index));
            if (executable != null) api.releaseExecutable(executable);
        }
    }

    private static List<EvidenceCase> baseCases(DataType type, byte[] words) {
        var reshape = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.RESHAPE, 0, 1,
                        new long[] {256, 256})));
        var permute = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.permutation(0, 1, List.of(1, 0))));
        var contiguous = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.permutation(0, 1, List.of(1, 0)),
                MetalMpsGraphProgram.Node.contiguous(1, 2)));
        byte[] transposed = transpose(words, 256, 256);
        return List.of(
                new EvidenceCase("RESHAPE_EXHAUSTIVE", type, reshape,
                        List.of(canonical(type, 65536), affine(type, new long[] {256, 256},
                                new long[] {256, 1})),
                        new int[] {0}, 1, List.of(words), words),
                new EvidenceCase("PERMUTE_EXHAUSTIVE", type, permute,
                        List.of(canonical(type, 256, 256), affine(type,
                                new long[] {256, 256}, new long[] {1, 256})),
                        new int[] {0}, 1, List.of(words), transposed),
                new EvidenceCase("CONTIGUOUS_EXHAUSTIVE", type, contiguous,
                        List.of(canonical(type, 256, 256), affine(type,
                                        new long[] {256, 256}, new long[] {1, 256}),
                                canonical(type, 256, 256)),
                        new int[] {0}, 2, List.of(words), transposed));
    }

    private static List<EvidenceCase> expandedCases(DataType type, byte[] words) {
        var slice = new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.SLICE,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.SLICE,
                new long[] {1, 0, 65536, 0, 1})));
        var concat = new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.CONCAT,
                new int[] {0, 1}, new int[] {2},
                MetalMpsGraphProgram.AttributeKind.AXIS, new long[] {0})));
        var tile = new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.TILE,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.TILE, new long[] {1, 1})));
        return List.of(
                new EvidenceCase("SLICE_EXHAUSTIVE", type, slice,
                        List.of(canonical(type, 65536), materialized(
                                type, new long[] {65536}, new long[] {1})),
                        new int[] {0}, 1, List.of(words), words),
                new EvidenceCase("CONCAT_EXHAUSTIVE", type, concat,
                        List.of(canonical(type, 32768), canonical(type, 32768),
                                canonical(type, 65536)),
                        new int[] {0, 1}, 2,
                        List.of(slice(words, 0, 65536), slice(words, 65536, 131072)), words),
                new EvidenceCase("TILE_EXHAUSTIVE", type, tile,
                        List.of(canonical(type, 65536), canonical(type, 65536)),
                        new int[] {0}, 1, List.of(words), words));
    }

    private static List<EvidenceCase> oddCases(DataType type, byte[] words) {
        int count = words.length / Short.BYTES;
        var reshape = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.RESHAPE, 0, 1,
                        new long[] {1, count})));
        var permute = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.permutation(0, 1, List.of(1, 0))));
        var contiguous = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.permutation(0, 1, List.of(1, 0)),
                MetalMpsGraphProgram.Node.contiguous(1, 2)));
        var slice = new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.SLICE,
                new int[] {0}, new int[] {1}, MetalMpsGraphProgram.AttributeKind.SLICE,
                new long[] {1, 0, count, 0, 1})));
        var concat = new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.CONCAT,
                new int[] {0, 1}, new int[] {2},
                MetalMpsGraphProgram.AttributeKind.AXIS, new long[] {0})));
        var tile = new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.TILE,
                new int[] {0}, new int[] {1}, MetalMpsGraphProgram.AttributeKind.TILE,
                new long[] {1, 1})));
        return List.of(
                new EvidenceCase("RESHAPE_ODD", type, reshape,
                        List.of(canonical(type, count), affine(type,
                                new long[] {1, count}, new long[] {count, 1})),
                        new int[] {0}, 1, List.of(words), words),
                new EvidenceCase("PERMUTE_ODD", type, permute,
                        List.of(canonical(type, count, 1), affine(type,
                                new long[] {1, count}, new long[] {1, 1})),
                        new int[] {0}, 1, List.of(words), words),
                new EvidenceCase("CONTIGUOUS_ODD", type, contiguous,
                        List.of(canonical(type, count, 1), affine(type,
                                        new long[] {1, count}, new long[] {1, 1}),
                                canonical(type, 1, count)),
                        new int[] {0}, 2, List.of(words), words),
                new EvidenceCase("SLICE_ODD", type, slice,
                        List.of(canonical(type, count), materialized(
                                type, new long[] {count}, new long[] {1})),
                        new int[] {0}, 1, List.of(words), words),
                new EvidenceCase("CONCAT_ODD", type, concat,
                        List.of(canonical(type, 128), canonical(type, 129),
                                canonical(type, count)),
                        new int[] {0, 1}, 2,
                        List.of(slice(words, 0, 256), slice(words, 256, words.length)), words),
                new EvidenceCase("TILE_ODD", type, tile,
                        List.of(canonical(type, count), canonical(type, count)),
                        new int[] {0}, 1, List.of(words), words));
    }

    private static MetalMpsGraphProgram.ValueDescriptor canonical(
            DataType type, long... dimensions) {
        Shape shape = Shape.of(dimensions);
        return new MetalMpsGraphProgram.ValueDescriptor(
                type, dimensions, Optional.of(LayoutDescriptor.contiguous(shape)), false, false);
    }

    private static MetalMpsGraphProgram.ValueDescriptor affine(
            DataType type, long[] dimensions, long[] strides) {
        Shape shape = Shape.of(dimensions);
        return new MetalMpsGraphProgram.ValueDescriptor(
                type, dimensions, Optional.of(LayoutDescriptor.of(shape, strides, 0L, true)),
                false, true);
    }

    private static MetalMpsGraphProgram.ValueDescriptor materialized(
            DataType type, long[] dimensions, long[] strides) {
        Shape shape = Shape.of(dimensions);
        return new MetalMpsGraphProgram.ValueDescriptor(
                type, dimensions, Optional.of(LayoutDescriptor.of(shape, strides, 0L, true)),
                false, false);
    }

    private static byte[] exhaustiveWords() {
        return sequentialWords(EXHAUSTIVE_WORDS);
    }

    private static byte[] sequentialWords(int count) {
        byte[] result = new byte[Math.multiplyExact(count, Short.BYTES)];
        for (int word = 0; word < count; word++) {
            result[word * 2] = (byte) word;
            result[word * 2 + 1] = (byte) (word >>> 8);
        }
        return result;
    }

    private static byte[] transpose(byte[] source, int rows, int columns) {
        byte[] result = new byte[source.length];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                int from = (row * columns + column) * Short.BYTES;
                int to = (column * rows + row) * Short.BYTES;
                result[to] = source[from];
                result[to + 1] = source[from + 1];
            }
        }
        return result;
    }

    private static byte[] slice(byte[] source, int begin, int end) {
        return java.util.Arrays.copyOfRange(source, begin, end);
    }

    private static void writeEvidence(
            MetalCertificationEnvironment environment,
            List<Observation> observations,
            List<CertificateCase> certificates) throws IOException {
        String requested = System.getenv("SYNAPTIK_METAL_CERTIFICATE_EVIDENCE_DIR");
        if (requested == null || requested.isBlank()) return;
        Path directory = Path.of(requested).toAbsolutePath().normalize();
        Files.createDirectories(directory);
        String environmentLine = String.join("\t",
                environment.gpuFamily(), environment.osBuild(), environment.sdkVersion(),
                environment.compilerVersion(), environment.binaryDigest(),
                environment.flagsOptions(), environment.capabilityManifestHash());
        StringBuilder accuracy = new StringBuilder(
                "schema\tgpu-family\tos-build\tsdk-version\tcompiler-version\tbinary-digest"
                        + "\tflags-options\tcapability-manifest-hash\n")
                .append("1\t").append(environmentLine).append('\n')
                .append("case\tprofile\tdtype\tword-count\tprogram-digest\tverdict\n");
        StringBuilder determinism = new StringBuilder(
                "schema\tcontract\trepeats\n1\tREPEAT_3_IDENTICAL_V1\t3\n")
                .append("case\tprofile\tdtype\tword-count\tprogram-digest\trepeats\tverdict\n");
        for (Observation observation : observations) {
            accuracy.append(observation.accuracyLine());
            determinism.append(observation.determinismLine());
        }
        Path accuracyPath = directory.resolve("accuracy.tsv");
        Path determinismPath = directory.resolve("determinism.tsv");
        Files.writeString(accuracyPath, accuracy, StandardCharsets.UTF_8);
        Files.writeString(determinismPath, determinism, StandardCharsets.UTF_8);
        String accuracyDigest = sha256(Files.readAllBytes(accuracyPath));
        String determinismDigest = sha256(Files.readAllBytes(determinismPath));

        StringBuilder rows = new StringBuilder(MetalLowPrecisionCertificateStore.header())
                .append('\n');
        for (CertificateCase certificateCase : certificates) {
            EvidenceCase evidenceCase = certificateCase.evidenceCase();
            List<String> dtypes = new ArrayList<>();
            for (int ignored : evidenceCase.feeds()) dtypes.add(certificateCase.type().name());
            dtypes.add(certificateCase.type().name());
            LowPrecisionCertificateKey key = new LowPrecisionCertificateKey(
                    TraceNumericalProfile.valueOf(certificateCase.profile().name()),
                    MetalLowPrecisionCertificateStore.FAMILY,
                    dtypes,
                    LowPrecisionCertificateKey.NO_ACCUMULATOR,
                    MetalLowPrecisionCertificateStore.DOMAIN_PREFIX
                            + certificateCase.programDigest(),
                    MetalLowPrecisionCertificateStore.ROUTE,
                    environment.gpuFamily(), environment.osBuild(), environment.sdkVersion(),
                    environment.compilerVersion(), environment.binaryDigest(),
                    certificateCase.programDigest(), environment.flagsOptions(),
                    environment.capabilityManifestHash());
            rows.append(String.join("\t",
                    "1", certificateCase.profile().name(),
                    MetalLowPrecisionCertificateStore.FAMILY,
                    String.join(",", dtypes), LowPrecisionCertificateKey.NO_ACCUMULATOR,
                    MetalLowPrecisionCertificateStore.DOMAIN_PREFIX
                            + certificateCase.programDigest(),
                    MetalLowPrecisionCertificateStore.ROUTE,
                    environment.gpuFamily(), environment.osBuild(), environment.sdkVersion(),
                    environment.compilerVersion(), environment.binaryDigest(),
                    certificateCase.programDigest(), environment.flagsOptions(),
                    environment.capabilityManifestHash(),
                    MetalLowPrecisionCertificateStore.ENVELOPE, "CERTIFIED_ENVELOPE",
                    accuracyDigest, "PASS", key.digest(),
                    MetalLowPrecisionCertificateStore.DETERMINISM_CONTRACT,
                    determinismDigest, "PASS"))
                    .append('\n');
        }
        Files.writeString(directory.resolve("certificates.tsv"), rows, StandardCharsets.UTF_8);
    }

    private static String sha256(byte[] bytes) {
        try {
            return HexFormat.of().formatHex(
                    MessageDigest.getInstance("SHA-256").digest(bytes));
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }

    private record EvidenceCase(
            String name,
            DataType type,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int target,
            List<byte[]> feedBytes,
            byte[] expected) {
        EvidenceCase {
            values = List.copyOf(values);
            feeds = feeds.clone();
            feedBytes = List.copyOf(feedBytes);
            expected = expected.clone();
        }
        @Override public int[] feeds() { return feeds.clone(); }
    }

    private record Observation(
            String name,
            NumericalProfile profile,
            DataType type,
            int wordCount,
            String programDigest,
            String verdict,
            int repeats) {
        String accuracyLine() {
            return String.join("\t", name, profile.name(), type.name(),
                    Integer.toString(wordCount), programDigest, verdict) + '\n';
        }
        String determinismLine() {
            return String.join("\t", name, profile.name(), type.name(),
                    Integer.toString(wordCount), programDigest,
                    Integer.toString(repeats), verdict) + '\n';
        }
    }

    private record CertificateCase(
            NumericalProfile profile,
            DataType type,
            EvidenceCase evidenceCase,
            String programDigest) {
    }
}

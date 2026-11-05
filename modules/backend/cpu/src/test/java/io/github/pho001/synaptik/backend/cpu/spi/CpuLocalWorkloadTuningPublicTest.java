package io.github.pho001.synaptik.backend.cpu.spi;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.cpu.CpuLocalWorkloadTuning;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Distinct-package checks for the supported opaque CPU tuning collaboration. */
final class CpuLocalWorkloadTuningPublicTest {

    @Test
    void retainedCollaborationRejectsClosedUseAndValidNonTunableWork() {
        CpuBackendIntegration integration = CpuBackendIntegration.open();
        CpuLocalWorkloadTuning tuning = integration.localWorkloadTuning();
        CompileArtifacts artifacts = compile(integration, leaf(Shape.of(4)).contiguous());
        assertAll(
                () -> assertSame(tuning, integration.localWorkloadTuning()),
                () -> assertTrue(tuning.candidateHandoff(context(integration, artifacts)).isEmpty()));
        integration.close();
        integration.close();
        assertAll(
                () -> assertThrows(IllegalStateException.class,
                        integration::localWorkloadTuning),
                () -> assertThrows(IllegalStateException.class,
                        () -> tuning.candidateHandoff(context(integration, artifacts))));
    }

    @Test
    void ordinaryBindableBatchRoundTripsExactCandidateDecisions() {
        try (CpuBackendIntegration integration = CpuBackendIntegration.open()) {
            CpuLocalWorkloadTuning tuning = integration.localWorkloadTuning();
            CompileArtifacts artifacts = compile(integration,
                    leaf(Shape.of(2, 3)).matmul(leaf(Shape.of(3, 2))));
            assertEquals(2, artifacts.constants().bindableInputs().size());
            assertTrue(artifacts.constants().constantSources().isEmpty());
            var handoff = tuning.candidateHandoff(context(integration, artifacts)).orElseThrow();
            var batch = handoff.candidateBatch();
            var candidates = tuning.candidates(batch);
            assertFalse(candidates.isEmpty());
            assertSame(artifacts.partitions().getFirst(), handoff.partition());
            assertTrue(handoff.selectedDecision().isEmpty());
            var repeatedBatch = tuning.candidateHandoff(
                    context(integration, artifacts)).orElseThrow().candidateBatch();
            assertAll(
                    () -> assertEquals(tuning.compatibility(batch),
                            tuning.compatibility(repeatedBatch)));


            var compatibility = tuning.compatibility(batch);
            byte[] compatibilityBytes = compatibility.bytes();
            compatibilityBytes[0] ^= 1;
            assertFalse(Arrays.equals(compatibilityBytes, compatibility.bytes()));
            var firstIdentity = tuning.candidateIdentity(candidates.getFirst());
            byte[] identityBytes = firstIdentity.bytes();
            identityBytes[0] ^= 1;
            assertFalse(Arrays.equals(identityBytes, firstIdentity.bytes()));

            for (var candidate : candidates) {
                var decision = tuning.selectedDecision(batch, candidate);
                byte[] encoded = tuning.encodeDecision(decision);
                var decoded = tuning.decodeCompatibleDecision(batch, encoded).orElseThrow();
                assertEquals(decision, decoded);
                byte[] corrupt = encoded.clone();
                corrupt[corrupt.length - 1] ^= 1;
                assertTrue(tuning.decodeCompatibleDecision(batch, corrupt).isEmpty());
                assertTrue(tuning.decodeCompatibleDecision(batch,
                        Arrays.copyOf(encoded, encoded.length - 1)).isEmpty());
                assertTrue(tuning.decodeCompatibleDecision(batch,
                        Arrays.copyOf(encoded, encoded.length + 1)).isEmpty());
            }

            var secondBatch = tuning.candidateHandoff(context(integration, artifacts))
                    .orElseThrow().candidateBatch();
            assertThrows(IllegalArgumentException.class,
                    () -> tuning.selectedDecision(secondBatch, candidates.getFirst()));
            var firstDecision = tuning.selectedDecision(batch, candidates.getFirst());
            assertThrows(IllegalArgumentException.class,
                    () -> tuning.selectedPreparation(secondBatch, firstDecision));
            for (int staleVersion : List.of(2, 3)) {
                byte[] stale = tuning.encodeDecision(firstDecision);
                java.nio.ByteBuffer.wrap(stale).putInt(4, staleVersion);
                assertTrue(tuning.decodeCompatibleDecision(batch, stale).isEmpty());
            }

        }
    }

    @Test
    void everyOrdinaryBindableCandidateMatchesIndependentVariedMatmulOracle() throws Exception {
        try (CpuBackendIntegration integration = CpuBackendIntegration.open()) {
            for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
                for (boolean transposed : List.of(false, true)) {
                    exerciseEveryMatmulCandidate(integration, type, transposed);
                }
            }
        }
    }

    @Test
    void everyConfiguredOneDnnCandidateMatchesIndependentVariedMatmulOracle() throws Exception {
        String library = System.getenv("SYNAPTIK_ONEDNN_LIBRARY");
        org.junit.jupiter.api.Assumptions.assumeTrue(library != null
                && Files.isRegularFile(Path.of(library)),
                "requires the established exact oneDNN 3.14.0 LP64 SEQ test library");
        for (var api : CpuBackendIntegration.OneDnnApi.values()) {
            try (var integration = CpuBackendIntegration.openOneDnn(Path.of(library), api)) {
                for (boolean transposed : List.of(false, true)) {
                    exerciseEveryMatmulCandidate(integration, DataType.FLOAT32, transposed);
                }
            }
        }
    }

    @Test
    void everyCompleteCandidateReadsActualBroadcastOrStridedBiasCoordinates() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            int n = type == DataType.FLOAT32
                    ? jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.length()
                    : jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.length();
            for (int biasStride : new int[] {0, 2}) {
                Shape leftShape = Shape.of(1, 2), rightShape = Shape.of(2, n), biasShape = Shape.of(n);
                Tensor left = TensorFactory.create(new TensorDescriptor(type, leftShape,
                        Optional.of(LayoutDescriptor.contiguous(leftShape)), false));
                Tensor right = TensorFactory.create(new TensorDescriptor(type, rightShape,
                        Optional.of(LayoutDescriptor.contiguous(rightShape)), false));
                Tensor bias = TensorFactory.create(new TensorDescriptor(type, biasShape,
                        Optional.of(LayoutDescriptor.of(biasShape, new long[] {biasStride}, 0, true)), false));
                try (Arena arena = Arena.ofShared();
                        CpuBackendIntegration integration = CpuBackendIntegration.open()) {
                    CompileArtifacts artifacts = compile(integration, left.matmul(right).add(bias));
                    var projection = context(integration, artifacts);
                    assertTrue(integration.localWorkloadTuning().candidateHandoff(projection).isEmpty());
                    var complete = integration.completePlanTuning();
                    var batch = complete.candidateHandoff(projection, Optional.empty())
                            .orElseThrow().candidateBatch();
                    var candidates = complete.candidates(batch);
                    assertFalse(candidates.isEmpty(), "the numerical oracle must execute a real domain");
                    TensorDescriptor descriptor = artifacts.graph().values().stream()
                            .filter(value -> value.id().equals(artifacts.graph().outputs().getFirst()))
                            .findFirst().orElseThrow().descriptor();
                    int biasCount = 1 + (n - 1) * biasStride;
                    var leftStorage = nativeStorage(arena, type, 2, index -> index + 1);
                    var rightStorage = nativeStorage(arena, type, 2 * n, index -> index < n ? 1 : 2);
                    // Poison unreferenced cells: a contiguous load cannot accidentally pass.
                    var biasStorage = nativeStorage(arena, type, biasCount, index -> biasStride == 0
                            ? 7 : index % 2 == 0 ? index / 2 + 1 : 1000);
                    try (var leftInput = integration.borrow(leftStorage);
                            var rightInput = integration.borrow(rightStorage);
                            var biasInput = integration.borrow(biasStorage)) {
                        var inputs = artifacts.constants().bindableInputBindings().stream().map(binding ->
                                binding.tensorId().equals(left.id()) ? leftInput
                                        : binding.tensorId().equals(right.id()) ? rightInput : biasInput).toList();
                        for (int candidateIndex = 0; candidateIndex < candidates.size(); candidateIndex++) {
                            try (var prepared = GraphPreparation.prepare(artifacts,
                                    List.of(complete.trialPreparation(batch, candidates.get(candidateIndex))),
                                    integration.scheduleAssembler());
                                    var result = new PreparedExecutionRunner().run(prepared, inputs)) {
                                var bytes = ByteBuffer.wrap(integration.copyToCanonicalHostBytes(
                                        result.publicationRepresentation(0), descriptor,
                                        (long) n * type.byteWidth())).order(ByteOrder.BIG_ENDIAN);
                                for (int column = 0; column < n; column++) {
                                    double actual = type == DataType.FLOAT32 ? bytes.getFloat() : bytes.getDouble();
                                    assertEquals((double) (5 + (biasStride == 0 ? 7 : column + 1)), actual,
                                            type + " biasStride=" + biasStride + " candidate="
                                                    + candidateIndex + " column=" + column);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static void exerciseEveryMatmulCandidate(CpuBackendIntegration integration,
            DataType type, boolean transposed) throws Exception {
        int m = 3, k = 7, n = 5;
        Shape leftShape = transposed ? Shape.of(k, m) : Shape.of(m, k);
        Shape rightShape = transposed ? Shape.of(n, k) : Shape.of(k, n);
        Tensor left = TensorFactory.create(new TensorDescriptor(type, leftShape,
                Optional.of(LayoutDescriptor.contiguous(leftShape)), false));
        Tensor right = TensorFactory.create(new TensorDescriptor(type, rightShape,
                Optional.of(LayoutDescriptor.contiguous(rightShape)), false));
        Tensor output = (transposed ? left.permute(1, 0) : left)
                .matmul(transposed ? right.permute(1, 0) : right);
        CompileArtifacts artifacts = compile(integration, output);
        var projection = context(integration, artifacts);
        var local = integration.localWorkloadTuning();
        var handoff = local.candidateHandoff(projection);
        var preparations = new java.util.ArrayList<
                io.github.pho001.synaptik.prepare.PartitionPreparation<?, ?>>();
        if (handoff.isPresent()) {
            var batch = handoff.orElseThrow().candidateBatch();
            for (var candidate : local.candidates(batch)) {
                preparations.add(local.trialPreparation(batch, candidate));
            }
        } else {
            assertTrue(transposed, "ordinary contiguous MATMUL must expose local configurations");
            var complete = integration.completePlanTuning();
            var batch = complete.candidateHandoff(projection, Optional.empty())
                    .orElseThrow().candidateBatch();
            for (var candidate : complete.candidates(batch)) {
                preparations.add(complete.trialPreparation(batch, candidate));
            }
        }
        assertFalse(preparations.isEmpty());
        TensorDescriptor descriptor = artifacts.graph().values().stream()
                .filter(value -> value.id().equals(artifacts.graph().outputs().getFirst()))
                .findFirst().orElseThrow().descriptor();
        for (int candidateIndex = 0; candidateIndex < preparations.size(); candidateIndex++) {
            try (var prepared = GraphPreparation.prepare(artifacts,
                    List.of(preparations.get(candidateIndex)), integration.scheduleAssembler())) {
                for (int pass = 0; pass < 2; pass++) {
                    final int currentPass = pass;
                    var leftStorage = matrixStorage(type, m * k, index -> {
                        int row = transposed ? index % m : index / k;
                        int inner = transposed ? index / m : index % k;
                        return leftValue(row, inner, currentPass);
                    });
                    var rightStorage = matrixStorage(type, k * n, index -> {
                        int inner = transposed ? index % k : index / n;
                        int column = transposed ? index / k : index % n;
                        return rightValue(inner, column, currentPass);
                    });
                    try (var leftInput = integration.borrow(leftStorage);
                            var rightInput = integration.borrow(rightStorage)) {
                        var inputs = artifacts.constants().bindableInputBindings().stream()
                                .map(binding -> binding.tensorId().equals(left.id())
                                        ? leftInput : rightInput).toList();
                        try (var result = new PreparedExecutionRunner().run(prepared, inputs)) {
                            ByteBuffer bytes = ByteBuffer.wrap(integration.copyToCanonicalHostBytes(
                                    result.publicationRepresentation(0), descriptor,
                                    (long) m * n * type.byteWidth())).order(ByteOrder.BIG_ENDIAN);
                            for (int row = 0; row < m; row++) {
                                for (int column = 0; column < n; column++) {
                                    long expected = 0;
                                    for (int inner = 0; inner < k; inner++) {
                                        expected += (long) leftValue(row, inner, pass)
                                                * rightValue(inner, column, pass);
                                    }
                                    double actual = type == DataType.FLOAT32
                                            ? bytes.getFloat() : bytes.getDouble();
                                    assertEquals((double) expected, actual,
                                            "candidate=" + candidateIndex
                                                    + ", transposed=" + transposed + ", pass="
                                                    + pass + ", row=" + row + ", column=" + column);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static int leftValue(int row, int inner, int pass) {
        return row * 5 - inner * 2 + pass + 1;
    }

    private static int rightValue(int inner, int column, int pass) {
        return inner * 3 + column - pass * 2 - 4;
    }

    private static MemorySegmentStorage matrixStorage(DataType type, int count,
            java.util.function.IntUnaryOperator value) {
        if (type == DataType.FLOAT32) {
            float[] values = new float[count];
            for (int index = 0; index < count; index++) values[index] = value.applyAsInt(index);
            return new MemorySegmentStorage(type, count, MemorySegment.ofArray(values));
        }
        double[] values = new double[count];
        for (int index = 0; index < count; index++) values[index] = value.applyAsInt(index);
        return new MemorySegmentStorage(type, count, MemorySegment.ofArray(values));
    }

    private static MemorySegmentStorage nativeStorage(Arena arena, DataType type, int count,
            java.util.function.IntUnaryOperator value) {
        var segment = arena.allocate((long) count * type.byteWidth(), 8);
        if (type == DataType.FLOAT32) {
            for (int index = 0; index < count; index++)
                segment.setAtIndex(ValueLayout.JAVA_FLOAT, index, value.applyAsInt(index));
        } else {
            for (int index = 0; index < count; index++)
                segment.setAtIndex(ValueLayout.JAVA_DOUBLE, index, value.applyAsInt(index));
        }
        return new MemorySegmentStorage(type, count, segment);
    }

    private static io.github.pho001.synaptik.prepare.analysis.PrepareContext<?> context(
            CpuBackendIntegration integration, CompileArtifacts artifacts) {
        var preparation = integration.partitionPreparation();
        return GraphPreparation.project(artifacts, artifacts.partitions().getFirst(),
                preparation.backendInputs());
    }

    private static Tensor leaf(Shape shape) {
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false));
    }

    private static CompileArtifacts compile(CpuBackendIntegration integration, Tensor output) {
        return GraphCompilationPort.compile(CompileMode.FORWARD_ONLY, List.of(output), Optional.empty(), GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(), PartitionScoringConfig.neutral(), List.of(integration.capabilityProvider()), List.of(integration.availabilitySnapshot()));
    }

}

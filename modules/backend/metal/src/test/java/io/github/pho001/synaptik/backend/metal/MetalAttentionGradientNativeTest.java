package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.compiler.FunctionalGradientRequest;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorId;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/** Exercises Compiler-produced ordinary adjoints rather than handwritten native derivatives. */
class MetalAttentionGradientNativeTest {
    private static final float[] QUERY = {0.2f, -0.1f, 0.3f, 0.4f};
    private static final float[] KEY = {0.1f, 0.2f, -0.2f, 0.3f, 0.4f, -0.1f,
            0.3f, 0.2f, 0.1f, -0.4f, -0.1f, 0.2f};
    private static final float[] VALUE = {0.2f, 0.7f, -0.3f, 0.4f, 0.8f, -0.2f};
    private static final byte[] MASK = {1, 0, 1, 1, 1, 1};

    @Test
    void outputRootProducesEveryRoleAdjointWithBroadcastReductionAndSavedCanonicalWeights() {
        for (boolean masked : List.of(false, true)) {
            for (boolean explicit : List.of(false, true)) assertAttention(false, masked, explicit);
        }
    }

    @Test
    void canonicalWeightsRootProducesSelectedQueryAndKeyAdjoints() {
        for (boolean masked : List.of(false, true)) {
            for (boolean explicit : List.of(false, true)) assertAttention(true, masked, explicit);
        }
    }

    private static void assertAttention(boolean weightsRoot, boolean masked, boolean explicit) {
        String library = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(library != null && !library.isBlank(), "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        Tensor query = tensor(DataType.FLOAT32, true, 1, 2, 2);
        Tensor key = tensor(DataType.FLOAT32, true, 2, 3, 2);
        Tensor value = tensor(DataType.FLOAT32, true, 1, 3, 2);
        Tensor mask = tensor(DataType.BOOL, false, 2, 3);
        var attrs = new ScaledDotProductAttentionAttrs(
                explicit ? Optional.of(ScalarValue.float32(0.5f)) : Optional.empty(), masked);
        var attention = masked ? query.scaledDotProductAttentionWithWeights(key, value, mask, attrs)
                : query.scaledDotProductAttentionWithWeights(key, value, attrs);
        Tensor root = weightsRoot ? attention.weights() : attention.output();
        Tensor seed = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, root.descriptor().shape(),
                Optional.of(LayoutDescriptor.contiguous(root.descriptor().shape())), false));
        float[] seeds = new float[weightsRoot ? 12 : 8];
        for (int i = 0; i < seeds.length; i++) seeds[i] = (i % 5 - 2) * 0.25f;
        List<Tensor> roles = weightsRoot ? List.of(query, key) : List.of(query, key, value);
        var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.ForwardTensorReference(root)),
                List.of(Optional.of(seed)), roles, false,
                FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
        var availability = new BackendAvailabilitySnapshot(MetalCapabilityProvider.METAL_BACKEND_ID,
                Map.of(new BackendDeviceId(MetalCapabilityProvider.METAL_BACKEND_ID, "default"),
                        DeviceClass.ACCELERATOR));
        var artifacts = GraphCompilationPort.compile(CompileMode.FORWARD_AND_BACKWARD,
                List.of(root), Optional.of(request), GraphOptimizationConfig.disabled(),
                BackendIntent.unconstrained(), PartitionScoringConfig.neutral(),
                List.of(new MetalCapabilityProvider()), List.of(availability));
        assertEquals(1, artifacts.partitions().size(), "the actual primal/adjoint closure remains Metal-only");
        List<ValueId> publications = new ArrayList<>();
        artifacts.publication().forwardBindings().forEach(binding -> publications.add(binding.valueId()));
        artifacts.publication().gradientBindings().forEach(binding -> publications.add(binding.valueId()));
        List<float[]> expected = oracle(weightsRoot, masked, explicit, seeds);
        assertEquals(expected.size(), publications.size());
        try (MetalDeviceContext context = MetalDeviceContext.open(Path.of(library));
                Arena arena = Arena.ofConfined()) {
            var analyzed = new AtomicReference<MetalNegPreparationPlan>();
            var preparer = new MetalNegPartitionPreparer();
            var preparation = new PartitionPreparation<>(new MetalNegAnalysisInputs(context),
                    prepareContext -> {
                        var result = preparer.analyze(prepareContext);
                        analyzed.set(result.plan());
                        return result;
                    }, new MetalNegPartitionFinalizer(context));
            try (var execution = GraphPreparation.prepare(artifacts, List.of(preparation),
                    scheduleContext -> new MetalNegPreparedScheduleAssembler(
                            context, analyzed.get(), publications).assemble(scheduleContext))) {
                assertEquals(MetalPreparedRoute.CUSTOM_PROGRAM, analyzed.get().route());
                Map<TensorId, MetalBufferRepresentation> buffers = new HashMap<>();
                try {
                    putFloat(buffers, query, QUERY, context, arena);
                    putFloat(buffers, key, KEY, context, arena);
                    putFloat(buffers, value, VALUE, context, arena);
                    putFloat(buffers, seed, seeds, context, arena);
                    if (masked) {
                        var maskBuffer = context.createBuffer(MASK.length);
                        buffers.put(mask.id(), maskBuffer);
                        var host = arena.allocateFrom(java.lang.foreign.ValueLayout.JAVA_BYTE, MASK);
                        maskBuffer.upload(0, host, 0, host.byteSize());
                    }
                    var inputs = artifacts.constants().bindableInputBindings().stream()
                            .<BufferRepresentation>map(binding -> buffers.get(binding.tensorId())).toList();
                    try (var result = new PreparedExecutionRunner().run(execution, inputs)) {
                        for (int publication = 0; publication < expected.size(); publication++) {
                            float[] wanted = expected.get(publication);
                            var host = arena.allocate((long) wanted.length * Float.BYTES, Float.BYTES);
                            ((MetalBufferRepresentation) result.publicationRepresentation(publication))
                                    .download(0, host, 0, host.byteSize());
                            assertArrayEquals(wanted, host.toArray(JAVA_FLOAT), 3e-5f,
                                    "publication " + publication + ", weightsRoot=" + weightsRoot
                                            + ", mask/causal=" + masked + ", explicit=" + explicit);
                        }
                    }
                } finally {
                    for (var buffer : buffers.values()) buffer.close();
                }
            }
        }
    }

    private static Tensor tensor(DataType type, boolean grad, long... dimensions) {
        Shape shape = Shape.of(dimensions);
        return TensorFactory.create(new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), grad));
    }

    private static void putFloat(Map<TensorId, MetalBufferRepresentation> buffers,
            Tensor tensor, float[] values, MetalDeviceContext context, Arena arena) {
        var buffer = context.createBuffer((long) values.length * Float.BYTES);
        buffers.put(tensor.id(), buffer);
        var host = arena.allocateFrom(JAVA_FLOAT, values);
        buffer.upload(0, host, 0, host.byteSize());
    }

    /** Independent scalar formula with explicit broadcast accumulation for each original role. */
    private static List<float[]> oracle(boolean weightsRoot, boolean masked, boolean explicit, float[] seed) {
        float scale = explicit ? 0.5f : (float) (1.0 / Math.sqrt(2));
        float[] output = new float[weightsRoot ? 12 : 8];
        float[] dq = new float[QUERY.length], dk = new float[KEY.length], dv = new float[VALUE.length];
        for (int batch = 0; batch < 2; batch++) {
            for (int i = 0; i < 2; i++) {
                int row = batch * 2 + i;
                float[] weights = new float[3];
                float sum = 0;
                for (int j = 0; j < 3; j++) {
                    if (masked && (j > i || MASK[i * 3 + j] == 0)) continue;
                    float score = 0;
                    for (int e = 0; e < 2; e++)
                        score += QUERY[i * 2 + e] * KEY[(batch * 3 + j) * 2 + e];
                    weights[j] = (float) Math.exp(score * scale);
                    sum += weights[j];
                }
                for (int j = 0; j < 3; j++) weights[j] /= sum;
                float[] upstream = new float[3];
                for (int j = 0; j < 3; j++) {
                    if (weightsRoot) {
                        output[row * 3 + j] = weights[j];
                        upstream[j] = seed[row * 3 + j];
                    } else {
                        for (int d = 0; d < 2; d++) {
                            output[row * 2 + d] += weights[j] * VALUE[j * 2 + d];
                            upstream[j] += seed[row * 2 + d] * VALUE[j * 2 + d];
                            dv[j * 2 + d] += weights[j] * seed[row * 2 + d];
                        }
                    }
                }
                float center = 0;
                for (int j = 0; j < 3; j++) center += weights[j] * upstream[j];
                for (int j = 0; j < 3; j++) {
                    float ds = weights[j] * (upstream[j] - center) * scale;
                    for (int e = 0; e < 2; e++) {
                        dq[i * 2 + e] += ds * KEY[(batch * 3 + j) * 2 + e];
                        dk[(batch * 3 + j) * 2 + e] += ds * QUERY[i * 2 + e];
                    }
                }
            }
        }
        return weightsRoot ? List.of(output, dq, dk) : List.of(output, dq, dk, dv);
    }
}

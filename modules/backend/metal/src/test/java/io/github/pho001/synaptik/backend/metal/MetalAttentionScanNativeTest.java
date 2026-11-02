package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorId;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/** Actual length-aware native Scan checks for independently shaped attention siblings. */
class MetalAttentionScanNativeTest {
    @Test
    void emptyPrimaryFeaturesDoNotSuppressNonemptyRowWeightsOrReadInactiveNaNQueries() {
        String library = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(library != null && !library.isBlank(), "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        for (boolean reverse : List.of(false, true)) {
            Tensor query = tensor(DataType.FLOAT32, 2, 1);
            Tensor key = tensor(DataType.FLOAT32, 2, 1);
            Tensor value = tensor(DataType.FLOAT32, 2, 0);
            var attention = query.scaledDotProductAttentionWithWeights(key, value);
            Tensor sequence = tensor(DataType.FLOAT32, 2, 2, 1);
            Tensor actualKey = tensor(DataType.FLOAT32, 2, 1);
            Tensor actualValue = tensor(DataType.FLOAT32, 2, 0);
            Tensor lengths = tensor(DataType.INT64, 2);
            var body = new ScanBody(List.of(), List.of(query), List.of(key, value), List.of(),
                    List.of(attention.output(), attention.weights()));
            var scan = Scan.apply(List.of(), List.of(sequence), List.of(actualKey, actualValue),
                    lengths, 0, List.of(OptionalInt.of(0)), body, reverse);
            var availability = new BackendAvailabilitySnapshot(MetalCapabilityProvider.METAL_BACKEND_ID,
                    Map.of(new BackendDeviceId(MetalCapabilityProvider.METAL_BACKEND_ID, "default"),
                            DeviceClass.ACCELERATOR));
            var artifacts = GraphCompilationPort.compile(CompileMode.FORWARD_ONLY, scan.outputs(), Optional.empty(),
                    GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(), PartitionScoringConfig.neutral(),
                    List.of(new MetalCapabilityProvider()), List.of(availability));
            assertEquals(1, artifacts.partitions().size(), "the complete bounded Scan remains on Metal");
            List<ValueId> publications = new ArrayList<>();
            artifacts.publication().forwardBindings().forEach(binding -> publications.add(binding.valueId()));
            assertEquals(2, publications.size());
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
                    Map<TensorId, MetalBufferRepresentation> buffers = new HashMap<>();
                    try {
                        put(buffers, sequence, arena.allocateFrom(JAVA_FLOAT, 0, 0, 0, Float.NaN), context);
                        put(buffers, actualKey, arena.allocateFrom(JAVA_FLOAT, 0, 0), context);
                        put(buffers, actualValue, arena.allocate(0, 1), context);
                        put(buffers, lengths, arena.allocateFrom(JAVA_LONG, 2, 1), context);
                        var inputs = artifacts.constants().bindableInputBindings().stream()
                                .<BufferRepresentation>map(binding -> buffers.get(binding.tensorId())).toList();
                        try (var result = new PreparedExecutionRunner().run(execution, inputs)) {
                            assertEquals(0, ((MetalBufferRepresentation) result.publicationRepresentation(0)).byteSize());
                            var weights = (MetalBufferRepresentation) result.publicationRepresentation(1);
                            var host = arena.allocate(8L * Float.BYTES, Float.BYTES);
                            weights.download(0, host, 0, host.byteSize());
                            assertArrayEquals(new float[] {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0, 0},
                                    host.toArray(JAVA_FLOAT), 0,
                                    "weight row activity is independent of primary feature cardinality");
                        }
                    } finally {
                        for (var buffer : buffers.values()) buffer.close();
                    }
                }
            }
        }
    }

    private static Tensor tensor(DataType type, long... dimensions) {
        Shape shape = Shape.of(dimensions);
        return TensorFactory.create(new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false));
    }

    private static void put(Map<TensorId, MetalBufferRepresentation> buffers, Tensor tensor,
            MemorySegment host, MetalDeviceContext context) {
        var buffer = context.createBuffer(host.byteSize());
        buffers.put(tensor.id(), buffer);
        if (host.byteSize() != 0) buffer.upload(0, host, 0, host.byteSize());
    }
}

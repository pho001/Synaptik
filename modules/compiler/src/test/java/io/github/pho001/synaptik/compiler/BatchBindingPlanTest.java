package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.contract.*;
import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.planning.capability.*;
import java.util.*;
import org.junit.jupiter.api.Test;

class BatchBindingPlanTest {
    private static final DynamicDimension B = new DynamicDimension("B");

    @Test void preservesOffsetOfFixedSourceInSymbolicExpandedPublication() {
        Tensor x = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, Shape.ofDimensions(B), Optional.empty(), false));
        Shape one = Shape.of(1);
        var sourceDescriptor = new TensorDescriptor(DataType.FLOAT32, one, Optional.of(LayoutDescriptor.of(one, new long[]{1}, 3, true)), false);
        Tensor scalar = TensorFactory.create(sourceDescriptor), expanded = scalar.expand(Shape.ofDimensions(B));
        var artifacts = compile(List.of(x.sum(), expanded)); var plan = BatchBindingPlan.prepare(artifacts);
        var actual = artifacts.constants().bindableInputBindings().stream().map(input -> input.tensorId().equals(x.id())
                ? descriptor(Shape.of(2)) : sourceDescriptor).toList();
        var binding = plan.bind(actual);
        var id = artifacts.publication().forwardBindings().get(1).valueId();
        int position = java.util.stream.IntStream.range(0, artifacts.graph().values().size()).filter(i -> artifacts.graph().values().get(i).id().equals(id)).findFirst().orElseThrow();
        var output = binding.descriptor(position);
        assertEquals(3, output.layout().orElseThrow().storageOffset());
        assertArrayEquals(new long[]{0}, output.layout().orElseThrow().strides());
        assertEquals(4, output.layout().orElseThrow().referencedElementSpan());
    }

    @Test void rejectsNegativeDerivedNonInputExtentDuringBindWithoutResolvingDescriptors() {
        Tensor x = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, Shape.ofDimensions(B), Optional.empty(), false));
        Tensor scalar = TensorFactory.create(descriptor(Shape.scalar()));
        Tensor expanded = scalar.expand(Shape.ofDimensions(DimensionExpressions.addConstant(B, -1)));
        var artifacts = compile(List.of(x.sum(), expanded)); var plan = BatchBindingPlan.prepare(artifacts);
        var actual = artifacts.constants().bindableInputBindings().stream().map(input -> input.tensorId().equals(x.id())
                ? descriptor(Shape.of(0)) : scalar.descriptor()).toList();
        assertThrows(IllegalArgumentException.class, () -> plan.bind(actual));
        var valid = artifacts.constants().bindableInputBindings().stream().map(input -> input.tensorId().equals(x.id())
                ? descriptor(Shape.of(2)) : scalar.descriptor()).toList();
        assertEquals(2, plan.bind(valid).batchSize());
    }

    @Test void validatesAllOccurrencesDenseReadViewsFlagsAndUnknownSymbols() {
        Tensor x = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, Shape.ofDimensions(B, new StaticDimension(4)), Optional.empty(), false));
        Tensor z = TensorFactory.create(x.descriptor()); var artifacts = compile(List.of(x.div(z))); var plan = BatchBindingPlan.prepare(artifacts);
        var dense = descriptor(Shape.of(40, 4));
        var view = new TensorDescriptor(DataType.FLOAT32, dense.shape(), Optional.of(LayoutDescriptor.of(dense.shape(), new long[]{4, 1}, 0, true)), false);
        assertEquals(40, plan.bind(List.of(dense, view)).batchSize());
        assertThrows(IllegalArgumentException.class, () -> plan.bind(List.of(dense, descriptor(Shape.of(1, 4)))));
        assertThrows(IllegalArgumentException.class, () -> plan.bind(List.of(dense, new TensorDescriptor(DataType.FLOAT32, dense.shape(), Optional.empty(), false))));
        assertThrows(IllegalArgumentException.class, () -> plan.bind(List.of(dense, new TensorDescriptor(DataType.FLOAT32, dense.shape(), dense.layout(), true))));
        assertThrows(NullPointerException.class, () -> plan.bind(Arrays.asList(dense, null)));
        assertThrows(IndexOutOfBoundsException.class, () -> plan.bind(List.of(dense, dense)).descriptor(Integer.MAX_VALUE));
        Tensor unknown = TensorFactory.create(descriptor(Shape.scalar())).expand(Shape.ofDimensions(new DynamicDimension("C")));
        assertThrows(IllegalArgumentException.class, () -> BatchBindingPlan.prepare(compile(List.of(x.sum(), unknown))));
    }

    @Test
    void preparesUnresolvedNamedAddOnceAndBindsDifferentActualBatches() {
        Shape declared = Shape.ofDimensions(B, new StaticDimension(4));
        Tensor left = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, declared, Optional.empty(), false));
        Tensor right = TensorFactory.create(left.descriptor());
        var artifacts = compile(List.of(left.add(right)));
        int outputPosition = outputPosition(artifacts, 0);
        TensorDescriptor symbolic = artifacts.graph().values().get(outputPosition).descriptor();
        assertEquals(declared, symbolic.shape());
        assertTrue(symbolic.layout().isEmpty());

        var plan = BatchBindingPlan.prepare(artifacts);
        var first = plan.bind(List.of(descriptor(Shape.of(3, 4)), descriptor(Shape.of(3, 4))));
        for (int batch : new int[] {0, 1, 7, 2}) {
            var actual = descriptor(Shape.of(batch, 4));
            var binding = plan.bind(List.of(actual, actual));
            assertEquals(batch, binding.batchSize());
            assertEquals(actual, binding.descriptor(outputPosition));
        }
        assertEquals(descriptor(Shape.of(3, 4)), first.descriptor(outputPosition));
        assertSame(symbolic, artifacts.graph().values().get(outputPosition).descriptor());
        assertTrue(symbolic.layout().isEmpty());
    }

    @Test
    void keepsStaticConvolutionClosureSeparateFromSymbolicMaterializationAdmission() {
        Tensor weight2d = TensorFactory.create(descriptor(Shape.of(2, 1, 1, 1)));
        Tensor weight3d = TensorFactory.create(descriptor(Shape.of(2, 1, 1, 1, 1)));
        Tensor image = TensorFactory.create(descriptor(Shape.of(3, 1, 2, 2)));
        Tensor volume = TensorFactory.create(descriptor(Shape.of(3, 1, 2, 2, 2)));
        var staticArtifacts = compile(List.of(
                image.conv2d(weight2d, Conv2dAttrs.defaults()),
                volume.conv3d(weight3d, Conv3dAttrs.defaults())));
        assertEquals(descriptor(Shape.of(3, 2, 2, 2)),
                staticArtifacts.graph().values().get(outputPosition(staticArtifacts, 0)).descriptor());
        assertEquals(descriptor(Shape.of(3, 2, 2, 2, 2)),
                staticArtifacts.graph().values().get(outputPosition(staticArtifacts, 1)).descriptor());
        for (var node : staticArtifacts.graph().nodes()) {
            if (node.operation().kind() instanceof Conv2dKind
                    || node.operation().kind() instanceof Conv3dKind) {
                assertFalse(ScanBody.hasCanonicalContiguousResults(node.operation()));
            }
        }

        Tensor namedImage = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                Shape.ofDimensions(B, new StaticDimension(1),
                        new StaticDimension(2), new StaticDimension(2)),
                Optional.empty(), false));
        Tensor namedVolume = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                Shape.ofDimensions(B, new StaticDimension(1),
                        new StaticDimension(2), new StaticDimension(2), new StaticDimension(2)),
                Optional.empty(), false));
        for (Tensor output : List.of(
                namedImage.conv2d(weight2d, Conv2dAttrs.defaults()),
                namedVolume.conv3d(weight3d, Conv3dAttrs.defaults()))) {
            var artifacts = compile(List.of(output));
            assertTrue(artifacts.graph().values().get(outputPosition(artifacts, 0))
                    .descriptor().layout().isEmpty());
            assertThrows(IllegalArgumentException.class, () -> BatchBindingPlan.prepare(artifacts));
        }
    }

    @Test
    void rejectsUnprovedNegativeAndEmptyExtractionsEvenWithADeclaredBatch() {
        Tensor batch = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.ofDimensions(B), Optional.empty(), false));
        Tensor vector = TensorFactory.create(descriptor(Shape.of(3)));
        for (Tensor extraction : List.of(
                vector.sliceByLength(new long[] {2}, new long[] {3},
                        new int[] {0}, new long[] {-1}),
                vector.cropToShape(Shape.of(0), Shape.of(0)))) {
            var artifacts = compile(List.of(batch.add(batch), extraction));
            assertTrue(artifacts.graph().values().get(outputPosition(artifacts, 1))
                    .descriptor().layout().isEmpty());
            assertThrows(IllegalArgumentException.class, () -> BatchBindingPlan.prepare(artifacts));
        }
    }

    private static int outputPosition(CompileArtifacts artifacts, int output) {
        var id = artifacts.publication().forwardBindings().get(output).valueId();
        return java.util.stream.IntStream.range(0, artifacts.graph().values().size())
                .filter(index -> artifacts.graph().values().get(index).id().equals(id))
                .findFirst().orElseThrow();
    }

    static TensorDescriptor descriptor(Shape shape) { return new TensorDescriptor(DataType.FLOAT32, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false); }
    static CompileArtifacts compile(List<Tensor> outputs) {
        BackendId id = new BackendId("binding-test");
        BackendCapabilityProvider provider = new BackendCapabilityProvider() {
            public BackendId backendId() { return id; }
            public boolean supports(OperationCapabilityQuery query) { return true; }
        };
        return GraphCompilationPort.compile(CompileMode.FORWARD_ONLY, outputs, Optional.empty(), GraphOptimizationConfig.standard(),
                BackendIntent.unconstrained(), PartitionScoringConfig.neutral(), List.of(provider),
                List.of(new BackendAvailabilitySnapshot(id, Map.of(new BackendDeviceId(id, "test"), DeviceClass.CPU))));
    }
}

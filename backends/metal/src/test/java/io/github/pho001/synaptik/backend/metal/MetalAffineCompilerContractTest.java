package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.FunctionalGradientRequest;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class MetalAffineCompilerContractTest {
    private static final BackendId BACKEND_ID = new BackendId("affine-contract");

    @Test
    void compilerRetainsForwardViewsAndBuildsExactFirstOrderAffineRules() {
        assertContract(
                tensor(Shape.of(2, 3), true),
                source -> source.reshape(3, 2),
                ShapeTransformKind.RESHAPE,
                new TargetShapeAttrs(Shape.of(3, 2)),
                ShapeTransformKind.RESHAPE,
                new TargetShapeAttrs(Shape.of(2, 3)));
        assertContract(
                tensor(Shape.of(1, 3), true),
                source -> source.expand(2, 3),
                ShapeTransformKind.EXPAND,
                new TargetShapeAttrs(Shape.of(2, 3)),
                AggregateReductionKind.SUM,
                new SumToShapeAttrs(Shape.of(1, 3)));
        assertContract(
                tensor(Shape.of(2, 3, 4), true),
                source -> source.permute(2, 0, 1),
                AxisTransformKind.PERMUTE,
                new PermutationAttrs(List.of(2, 0, 1)),
                AxisTransformKind.PERMUTE,
                new PermutationAttrs(List.of(1, 2, 0)));
        assertContract(
                tensor(Shape.of(2, 3), true),
                source -> source.expandDims(1),
                AxisTransformKind.EXPAND_DIMS,
                new AxisTransformAttrs(1),
                AxisTransformKind.SQUEEZE,
                new AxisTransformAttrs(1));
        assertContract(
                tensor(Shape.of(2, 1, 3), true),
                source -> source.squeeze(1),
                AxisTransformKind.SQUEEZE,
                new AxisTransformAttrs(1),
                AxisTransformKind.EXPAND_DIMS,
                new AxisTransformAttrs(1));
    }

    private static void assertContract(
            Tensor input,
            java.util.function.UnaryOperator<Tensor> transform,
            OperationKind forwardKind,
            OperationAttrs forwardAttrs,
            OperationKind backwardKind,
            OperationAttrs backwardAttrs) {
        Tensor output = transform.apply(input);
        Tensor seed = tensor(output.descriptor().shape(), false);
        FunctionalGradientRequest request = new FunctionalGradientRequest(List.of(
                new FunctionalGradientRequest.Stage(
                        List.of(new FunctionalGradientRequest.ForwardTensorReference(output)),
                        List.of(Optional.of(seed)),
                        List.of(input),
                        false,
                        FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
        CompileArtifacts artifacts = GraphCompilationPort.compile(CompileMode.FORWARD_AND_BACKWARD, io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, List.of(output), Optional.of(request), GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(), PartitionScoringConfig.neutral(), List.of(provider()), List.of(snapshot()));

        CompiledNode forward = node(artifacts, 0, forwardKind);
        assertEquals(new Operation(forwardKind, forwardAttrs), forward.operation());
        assertEquals(output.descriptor(), value(artifacts, forward).descriptor());

        CompiledNode backward = node(artifacts, 1, backwardKind);
        assertEquals(new Operation(backwardKind, backwardAttrs), backward.operation());
        TensorDescriptor backwardOutput = value(artifacts, backward).descriptor();
        assertEquals(input.descriptor().shape(), backwardOutput.shape());
        assertEquals(
                backwardLayout(seed, backwardKind, backwardAttrs),
                backwardOutput.layout().orElseThrow());
    }

    private static LayoutDescriptor backwardLayout(
            Tensor seed, OperationKind kind, OperationAttrs attrs) {
        if (attrs instanceof SumToShapeAttrs sumTo) {
            return LayoutDescriptor.contiguous(sumTo.targetShape());
        }
        Tensor affine;
        if (kind == ShapeTransformKind.RESHAPE) {
            affine = seed.reshape(((TargetShapeAttrs) attrs).targetShape().toLongArray());
        } else if (kind == AxisTransformKind.PERMUTE) {
            int[] axes = ((PermutationAttrs) attrs).axes().stream()
                    .mapToInt(Integer::intValue)
                    .toArray();
            affine = seed.permute(axes);
        } else if (kind == AxisTransformKind.EXPAND_DIMS) {
            affine = seed.expandDims(((AxisTransformAttrs) attrs).axis());
        } else if (kind == AxisTransformKind.SQUEEZE) {
            affine = seed.squeeze(((AxisTransformAttrs) attrs).axis());
        } else {
            throw new AssertionError(kind);
        }
        return affine.descriptor().layout().orElseThrow();
    }

    private static CompiledNode node(
            CompileArtifacts artifacts, int derivativeOrder, OperationKind kind) {
        return artifacts.graph().nodes().stream()
                .filter(node -> artifacts.derivatives().derivativeOrderByNode()
                        .get(node.id()) == derivativeOrder)
                .filter(node -> node.operation().kind() == kind)
                .findFirst()
                .orElseThrow();
    }

    private static GraphValue value(CompileArtifacts artifacts, CompiledNode node) {
        return artifacts.graph().values().stream()
                .filter(value -> value.id().equals(node.outputs().getFirst()))
                .findFirst()
                .orElseThrow();
    }

    private static Tensor tensor(Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                requiresGrad));
    }

    private static BackendCapabilityProvider provider() {
        return new BackendCapabilityProvider() {
            @Override
            public BackendId backendId() {
                return BACKEND_ID;
            }

            @Override
            public boolean supports(
                    io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery query) {
                return true;
            }
        };
    }

    private static BackendAvailabilitySnapshot snapshot() {
        BackendDeviceId deviceId = new BackendDeviceId(BACKEND_ID, "0");
        return new BackendAvailabilitySnapshot(
                BACKEND_ID, Map.of(deviceId, DeviceClass.CPU));
    }
}

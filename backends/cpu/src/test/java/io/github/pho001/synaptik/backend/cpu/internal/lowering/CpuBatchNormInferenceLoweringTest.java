package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormInferenceAttrs;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

public class CpuBatchNormInferenceLoweringTest {
    @Test void derivesArbitraryAxisCountsAndUniqueBoundariesWithoutResources() {
        var lowered = new CpuPartitionLowering().lower(context(
                List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.FLOAT64,
                        DataType.FLOAT64, DataType.FLOAT64), Shape.of(2, 3, 4), 1,
                List.of(0, 1, 1, 1, 1)));
        var ir = (CpuBatchNormInferenceIr) lowered.portableKernelIr();
        var geometry = lowered.batchNormInferenceGeometry().orElseThrow();
        assertAll(
                () -> assertEquals(List.of(0, 1, 1, 1, 1), ir.positionToBoundary()),
                () -> assertEquals(3, lowered.boundaryValues().size()),
                () -> assertEquals(2, geometry.prefixCount()),
                () -> assertEquals(3, geometry.channelCount()),
                () -> assertEquals(4, geometry.suffixCount()),
                () -> assertEquals(8, geometry.nonChannelCount()),
                () -> assertEquals(24, geometry.outputCount()),
                () -> assertTrue(lowered.virtualValues().isEmpty()),
                () -> assertTrue(lowered.trailingNormalizationGeometry().isEmpty()));
    }

    @Test void emptyChannelOrNonChannelDomainHasNoRange() {
        for (Shape shape : List.of(Shape.of(2, 0, 4), Shape.of(0, 3, 4), Shape.of(2, 3, 0))) {
            var lowered = new CpuPartitionLowering().lower(context(
                    java.util.Collections.nCopies(5, DataType.FLOAT32), shape, 1,
                    List.of(0, 1, 2, 3, 4)));
            assertEquals(0, lowered.elementCount());
            assertEquals(0, lowered.batchNormInferenceGeometry().orElseThrow().outputCount());
        }
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(List<DataType> types,
            Shape inputShape, int axis, List<Integer> occurrences) {
        Shape vector = Shape.of(inputShape.toLongArray()[axis]);
        var layouts = new ArrayList<LayoutDescriptor>();
        layouts.add(LayoutDescriptor.contiguous(inputShape));
        for (int index = 1; index < 5; index++) layouts.add(LayoutDescriptor.contiguous(vector));
        layouts.add(LayoutDescriptor.contiguous(inputShape));
        return context(types, inputShape, axis, occurrences, layouts);
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(List<DataType> types,
            Shape inputShape, int axis, List<Integer> occurrences,
            List<LayoutDescriptor> layouts) {
        if (types.size() != 5) throw new IllegalArgumentException("five types required");
        if (layouts.size() != 6) throw new IllegalArgumentException("six layouts required");
        Shape vector = Shape.of(inputShape.toLongArray()[axis]);
        DataType result = types.getFirst();
        for (int index = 1; index < types.size(); index++) result =
                io.github.pho001.synaptik.model.datatype.DataTypePromotion.promoteFloating(
                        result, types.get(index));
        ScalarValue epsilon = result == DataType.FLOAT64 ? ScalarValue.float64(1e-5)
                : result == DataType.FLOAT32 ? ScalarValue.float32(1e-5f)
                : ScalarValue.bfloat16Bits((short) 0x3728);
        var descriptors = new ArrayList<io.github.pho001.synaptik.model.tensor.TensorDescriptor>();
        descriptors.add(new TensorDescriptor(types.getFirst(), inputShape,
                java.util.Optional.of(layouts.getFirst()), false));
        for (int index = 1; index < 5; index++) {
            descriptors.add(new TensorDescriptor(types.get(index), vector,
                    java.util.Optional.of(layouts.get(index)), false));
        }
        return CpuScatterLoweringTest.context(new Operation(BatchNormKind.BATCH_NORM_INFERENCE,
                new BatchNormInferenceAttrs(axis, epsilon)), occurrences, descriptors,
                new TensorDescriptor(result, inputShape,
                        java.util.Optional.of(layouts.getLast()), false));
    }
}

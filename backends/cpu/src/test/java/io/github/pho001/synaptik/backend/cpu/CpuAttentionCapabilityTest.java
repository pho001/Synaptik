package io.github.pho001.synaptik.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuAttentionCapabilityTest {
  private static final List<DataType> TYPES =
      List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64);
  private final CpuCapabilityProvider provider = new CpuCapabilityProvider();

  @Test
  void acceptsAllOrderedFloatingTriplesAndExactMaskAndOutputForms() {
    for (DataType q : TYPES)
      for (DataType k : TYPES)
        for (DataType v : TYPES) {
          DataType result =
              DataTypePromotion.promoteFloating(DataTypePromotion.promoteFloating(q, k), v);
          for (boolean masked : List.of(false, true))
            for (int outputs : List.of(1, 2)) {
              var inputs =
                  new ArrayList<TensorDescriptor>(
                      List.of(
                          desc(q, Shape.of(2, 1, 3, 4), true),
                          desc(k, Shape.of(1, 5, 6, 4), true),
                          desc(v, Shape.of(2, 5, 6, 7), true)));
              if (masked) inputs.add(desc(DataType.BOOL, Shape.of(1, 5, 1, 6), false));
              var resultDescriptors = new ArrayList<TensorDescriptor>();
              resultDescriptors.add(desc(result, Shape.of(2, 5, 3, 7), true));
              if (outputs == 2) resultDescriptors.add(desc(result, Shape.of(2, 5, 3, 6), true));
              assertTrue(
                  provider.supports(query(inputs, resultDescriptors)), q + " " + k + " " + v);
            }
        }
  }

  @Test
  void rejectsMaskBroadcastGradientAndOutputLayoutMismatches() {
    var base =
        List.of(
            desc(DataType.FLOAT32, Shape.of(2, 3), false),
            desc(DataType.FLOAT32, Shape.of(4, 3), false),
            desc(DataType.FLOAT32, Shape.of(4, 5), false));
    var invalidMask = new ArrayList<>(base);
    invalidMask.add(desc(DataType.BOOL, Shape.of(2, 5), false));
    assertFalse(
        provider.supports(
            query(invalidMask, List.of(desc(DataType.FLOAT32, Shape.of(2, 5), false)))));
    var wrongGrad =
        new TensorDescriptor(
            DataType.FLOAT32,
            Shape.of(2, 5),
            Optional.of(LayoutDescriptor.contiguous(Shape.of(2, 5))),
            true);
    assertFalse(provider.supports(query(base, List.of(wrongGrad))));
    var overlapping =
        new TensorDescriptor(
            DataType.FLOAT32,
            Shape.of(2, 5),
            Optional.of(LayoutDescriptor.of(Shape.of(2, 5), new long[] {1, 1}, 0, false)),
            false);
    assertFalse(provider.supports(query(base, List.of(overlapping))));
  }

  private static OperationCapabilityQuery query(
      List<TensorDescriptor> inputs, List<TensorDescriptor> outputs) {
    return new OperationCapabilityQuery(
        new Operation(
            ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION,
            new ScaledDotProductAttentionAttrs(Optional.empty(), false)),
        inputs,
        outputs);
  }

  private static TensorDescriptor desc(DataType type, Shape shape, boolean grad) {
    return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), grad);
  }
}

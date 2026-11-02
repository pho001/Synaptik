package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

final class OrdinaryConstructionTest {
    @Test
    void metadataAndRealObserveTheSameEagerAttentionSitesIncludingUnusedValueCast() {
        Tensor query = tensor(DataType.FLOAT64, Shape.of(3, 4), true);
        Tensor key = tensor(DataType.FLOAT64, Shape.of(5, 4), true);
        Tensor value = tensor(DataType.FLOAT32, Shape.of(5, 6), true);
        var attention = query.scaledDotProductAttentionWithWeights(key, value);
        TensorProducer producer = attention.output().provenance().orElseThrow().producer();
        Tensor seed = tensor(DataType.FLOAT64, attention.weights().descriptor().shape(), false);
        List<Observed> realSites = new ArrayList<>();
        var real = OrdinaryConstruction.real(new FirstOrderAutograd.DerivativeConstants(),
                (operation, inputs, outputs, deferred) -> realSites.add(new Observed(operation,
                        inputs.stream().map(Tensor::descriptor).toList(),
                        outputs.stream().map(Tensor::descriptor).toList())));
        Tensor[] realResult = FirstOrderAutograd.applyOrdinary(
                FirstOrderGradientCoverage.FamilyOwner.ATTENTION,
                OrdinaryConstruction.source(producer), 1, seed,
                new boolean[] {true, false, false}, real, Function.identity());

        List<OrdinaryConstruction.Value> inputs = producer.inputs().stream()
                .map(t -> new OrdinaryConstruction.Value(t.descriptor())).toList();
        List<OrdinaryConstruction.Value> outputs = IntStream.range(0, producer.outputCount())
                .mapToObj(i -> new OrdinaryConstruction.Value(producer.output(i).descriptor())).toList();
        List<Observed> metadataSites = new ArrayList<>();
        IdentityHashMap<OrdinaryConstruction.Value, Boolean> resolved = new IdentityHashMap<>();
        var metadata = OrdinaryConstruction.metadata(null, null, null,
                (operation, operands, results, deferred) -> metadataSites.add(new Observed(operation,
                        operands.stream().map(OrdinaryConstruction.Value::descriptor).toList(),
                        results.stream().map(OrdinaryConstruction.Value::descriptor).toList())));
        Tensor marker = tensor(DataType.FLOAT32, Shape.scalar(), false);
        var source = OrdinaryConstruction.source(producer.operation(), inputs, outputs);
        OrdinaryConstruction.Value[] result = FirstOrderAutograd.applyOrdinary(
                FirstOrderGradientCoverage.FamilyOwner.ATTENTION, source, 1,
                new OrdinaryConstruction.Value(seed.descriptor()),
                new boolean[] {true, false, false}, metadata, original -> {
                    resolved.put(original, true);
                    return original;
                });
        assertEquals(marker.id().value() + 1,
                tensor(DataType.FLOAT32, Shape.scalar(), false).id().value());
        assertEquals(realSites, metadataSites);
        assertEquals(realResult[0].descriptor(), result[0].descriptor());
        assertNull(result[1]);
        assertNull(result[2]);
        assertTrue(resolved.containsKey(inputs.get(2)));
        assertTrue(resolved.containsKey(outputs.get(1)));
        assertFalse(resolved.containsKey(outputs.get(0)));
        assertTrue(metadataSites.stream().anyMatch(site -> site.operation().kind() == CastKind.CAST
                && site.inputs().getFirst().equals(value.descriptor())));
    }

    @Test
    void completeCanonicalSavedSiblingsAreCheckedWithoutConsumingIds() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        Tensor scale = tensor(DataType.FLOAT32, Shape.of(3), true);
        Tensor bias = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor mean = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor variance = tensor(DataType.FLOAT32, Shape.of(3), false);
        var result = input.batchNormTraining(1, scale, bias, mean, variance,
                ScalarValue.float32(0.1f), ScalarValue.float32(1.0e-5f));
        TensorProducer producer = result.output().provenance().orElseThrow().producer();
        List<OrdinaryConstruction.Value> operands = producer.inputs().stream()
                .map(t -> new OrdinaryConstruction.Value(t.descriptor())).toList();
        List<OrdinaryConstruction.Value> siblings = new ArrayList<>(IntStream.range(0, 5)
                .mapToObj(i -> new OrdinaryConstruction.Value(producer.output(i).descriptor())).toList());
        TensorDescriptor saved = siblings.get(4).descriptor();
        siblings.set(4, new OrdinaryConstruction.Value(new TensorDescriptor(
                DataType.FLOAT64, saved.shape(), saved.layout(), saved.requiresGrad())));
        Tensor marker = tensor(DataType.FLOAT32, Shape.scalar(), false);
        assertThrows(IllegalArgumentException.class,
                () -> OrdinaryConstruction.source(producer.operation(), operands, siblings));
        assertEquals(marker.id().value() + 1,
                tensor(DataType.FLOAT32, Shape.scalar(), false).id().value());
    }

    @Test
    void exactMetadataLiteralRolesDoNotResolvePrimalValuesOrAllocateTensorLeaves() {
        var construction = OrdinaryConstruction.metadata(null, null, null, null);
        var original = new OrdinaryConstruction.Value(new TensorDescriptor(
                DataType.FLOAT64, Shape.of(2, 3), Optional.empty(), true));
        Tensor marker = tensor(DataType.FLOAT32, Shape.scalar(), false);
        ScalarValue negativeZero = ScalarValue.float64(Double.longBitsToDouble(Long.MIN_VALUE));
        ScalarValue payload = ScalarValue.float64(Double.longBitsToDouble(0x7ff8000000000042L));
        var zero = construction.base(negativeZero);
        assertSame(zero, construction.base(negativeZero));
        assertNotSame(zero, construction.zeroBase(DataType.FLOAT64));
        assertEquals(Optional.of(negativeZero), zero.literal());
        assertEquals(Optional.of(payload), construction.base(payload).literal());
        assertEquals(original.descriptor().shape(),
                construction.valueLike(payload, original).descriptor().shape());
        assertEquals(marker.id().value() + 1,
                tensor(DataType.FLOAT32, Shape.scalar(), false).id().value());
    }

    @Test
    void scopedObservedReshapeGradientRejectsChangedSymbolicCoefficientBeforeDerivativeIds() {
        var batch = new DynamicDimension("ReshapeBatch");
        var inputShape = Shape.ofDimensions(batch, new StaticDimension(2));
        var changedShape = Shape.ofDimensions(batch, new StaticDimension(3));
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            Tensor input = tensor(type, inputShape, true);
            // Ordinary Tensor construction genuinely defers dynamic count equality. Do not
            // replace it with a ScanBody fixture, whose source guard already rejects this case.
            Tensor output = input.reshape(changedShape);
            assertEquals(changedShape, output.descriptor().shape());
            TensorProducer producer = output.provenance().orElseThrow().producer();
            var source = OrdinaryConstruction.source(producer.operation(),
                    producer.inputs().stream().map(t -> new OrdinaryConstruction.Value(t.descriptor())).toList(),
                    List.of(new OrdinaryConstruction.Value(output.descriptor())));
            var seed = new OrdinaryConstruction.Value(new TensorDescriptor(
                    type, changedShape, Optional.empty(), false));
            var ordinary = OrdinaryConstruction.metadata(null, null, null, null);
            var deferred = FirstOrderAutograd.applyOrdinary(
                    FirstOrderGradientCoverage.FamilyOwner.LAYOUT, source, 0, seed,
                    new boolean[] {true}, ordinary, Function.identity());
            assertEquals(inputShape, deferred[0].descriptor().shape());
            var scoped = OrdinaryConstruction.metadata(null, null, null,
                    (operation, inputs, outputs, predicates) -> ScanBodyInference.validateConstruction(
                            operation, inputs.stream().map(OrdinaryConstruction.Value::descriptor).toList(),
                            outputs.stream().map(OrdinaryConstruction.Value::descriptor).toList(),
                            predicates, Optional.of(batch)));
            Tensor marker = tensor(DataType.BOOL, Shape.scalar(), false);
            assertThrows(IllegalArgumentException.class, () -> FirstOrderAutograd.applyOrdinary(
                    FirstOrderGradientCoverage.FamilyOwner.LAYOUT, source, 0, seed,
                    new boolean[] {true}, scoped, Function.identity()));
            assertEquals(marker.id().value() + 1,
                    tensor(DataType.BOOL, Shape.scalar(), false).id().value(),
                    "scoped ordinary metadata rejection must precede derivative Tensor construction");
        }
    }

    @Test
    void metadataRetainsIndependentDeferredPredicatesAndCropLayoutCompatibility() {
        Shape n = Shape.ofDimensions(new DynamicDimension("N"));
        Shape m = Shape.ofDimensions(new DynamicDimension("M"));
        TensorDescriptor input = new TensorDescriptor(DataType.FLOAT64, n, Optional.empty(), false);
        Operation reshape = new Operation(ShapeTransformKind.RESHAPE, new TargetShapeAttrs(m));
        List<TensorDescriptor> outputs = ScanBody.describeOrdinary(reshape, List.of(input), 1);
        assertEquals(List.of("reshape element count"),
                CapturedGraphInference.checkOrdinary(reshape, List.of(input), outputs).stream()
                        .map(CapturedGraphInference.ConstraintRequest::subject).toList());
        assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.checkOrdinary(
                reshape, List.of(input), List.of(new TensorDescriptor(
                        DataType.FLOAT32, m, Optional.empty(), false))));
        var construction = OrdinaryConstruction.metadata(null, null, null, null);
        Tensor marker = tensor(DataType.FLOAT32, Shape.scalar(), false);
        var expanded = construction.expand(construction.oneBase(DataType.FLOAT64), Shape.of(4));
        var crop = construction.cropToShape(expanded, Shape.of(2), Shape.of(1));
        assertEquals(Shape.of(2), crop.descriptor().shape());
        assertTrue(crop.descriptor().layout().isEmpty());
        assertEquals(marker.id().value() + 1,
                tensor(DataType.FLOAT32, Shape.scalar(), false).id().value());
    }

    @Test
    void shapeLiteralBindingRequiresExplicitLiteralFactsBeforeConstruction() {
        var original = new OrdinaryConstruction.Value(new TensorDescriptor(
                DataType.FLOAT64, Shape.of(2, 3), Optional.empty(), true));
        ScalarValue value = ScalarValue.float64(-0.5d);
        var construction = OrdinaryConstruction.metadata(null, (literal, source) -> {
            return new OrdinaryConstruction.Value(new TensorDescriptor(
                    literal.dataType(), source.descriptor().shape(), Optional.empty(), false),
                    Optional.of(literal));
        }, source -> source == original ? 1 : -1, null);
        Tensor marker = tensor(DataType.FLOAT32, Shape.scalar(), false);
        var bound = construction.valueLike(value, original);
        assertEquals(original.descriptor().shape(), bound.descriptor().shape());
        assertFalse(bound.descriptor().requiresGrad());
        var missingFact = OrdinaryConstruction.metadata(null,
                (literal, source) -> new OrdinaryConstruction.Value(bound.descriptor()), null, null);
        assertThrows(IllegalArgumentException.class, () -> missingFact.valueLike(value, original));
        assertEquals(marker.id().value() + 1,
                tensor(DataType.FLOAT32, Shape.scalar(), false).id().value());
    }

    /**
     * Creates an unbacked ordinary Tensor for test setup or an allocation-boundary marker.
     * @param type exact requested data type
     * @param shape exact logical Shape
     * @param gradient whether the source requests gradients
     * @return a fresh storage-free Tensor
     */
    private static Tensor tensor(DataType type, Shape shape, boolean gradient) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.empty(), gradient));
    }

    /**
     * Records observable construction semantics without producer or graph-local identities.
     * @param operation exact ordinary operation
     * @param inputs ordered operand descriptors
     * @param outputs complete canonical output descriptors
     */
    private record Observed(Operation operation, List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {}
}

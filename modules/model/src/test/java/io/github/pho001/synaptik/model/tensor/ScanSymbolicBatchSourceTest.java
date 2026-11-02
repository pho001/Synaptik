package io.github.pho001.synaptik.model.tensor;

import static io.github.pho001.synaptik.model.tensor.ScanBodyTest.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.ordering.*;
import io.github.pho001.synaptik.model.operation.scan.*;
import io.github.pho001.synaptik.model.shape.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class ScanSymbolicBatchSourceTest {
    private static final DynamicDimension ROWS = new DynamicDimension("Rows");

    private static Shape row(long width) {
        return Shape.ofDimensions(ROWS, new StaticDimension(width));
    }

    private static Shape namedAt(int axis, long... sizes) {
        Dimension[] dims = new Dimension[sizes.length];
        for (int i = 0; i < sizes.length; i++) dims[i] = i == axis ? ROWS : new StaticDimension(sizes[i]);
        return Shape.ofDimensions(dims);
    }

    private static Shape stacked(long time, Shape shape) {
        var dims = new ArrayList<Dimension>();
        dims.add(new StaticDimension(time));
        dims.addAll(shape.dimensions());
        return Shape.ofDimensions(dims.toArray(Dimension[]::new));
    }

    private static Tensor source(DataType type, boolean grad, Shape shape) {
        return TensorFactory.create(new TensorDescriptor(type, shape, shape.isFullyStatic()
                ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(), grad));
    }

    private static Tensor source(boolean grad, Shape shape) {
        return source(DataType.FLOAT32, grad, shape);
    }

    private static void rejectsWithoutIds(Runnable action) throws Exception {
        long before = nextId();
        assertThrows(IllegalArgumentException.class, action::run);
        assertEquals(before, nextId());
    }

    @Test void preservesStaticAndSymbolicFormsWithoutChangingHeadersOrBindingBatch() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            var shapes = List.of(Shape.of(0, 4), Shape.of(1, 4), Shape.of(32, 4), row(4));
            for (Shape shape : shapes) for (long time : List.of(0L, 3L)) for (boolean reverse : List.of(false, true)) {
                Tensor h = source(type, false, shape), x = source(type, true, shape);
                Tensor shared = source(type, false, Shape.of(4));
                Tensor next = h.add(x).add(shared).tanh();
                long before = nextId();
                var body = body(h, x, List.of(shared), next, h, next, next, shared);
                assertEquals(before, nextId());
                assertEquals(List.of(h, next.provenance().orElseThrow().inputs().getFirst()
                        .provenance().orElseThrow().inputs().getFirst(),
                        next.provenance().orElseThrow().inputs().getFirst(), next), body.retainedValues());
                Tensor initial = source(type, false, shape), sequence = source(type, true, stacked(time, shape));
                Tensor parameter = source(type, false, Shape.of(4));
                before = nextId();
                var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(parameter), body, reverse);
                var p = scan.finalCarry().getFirst().provenance().orElseThrow().producer();
                assertEquals(new ScanAttrs(time, 1, 1, 1, 4, 4, reverse), p.operation().attrs());
                assertEquals(before + p.outputCount(), nextId());
                assertEquals(List.of(initial, sequence, parameter), p.inputs());
                assertSame(body, p.scanBody().orElseThrow());
                assertEquals(shape, scan.finalCarry().getFirst().descriptor().shape());
                assertNotSame(initial, scan.finalCarry().getFirst());
                assertNotSame(scan.outputs().get(1), scan.outputs().get(2));
                var roles = new ArrayList<>(body.nextCarry());
                roles.addAll(body.emissions()); roles.addAll(body.retainedValues());
                for (int i = 0; i < p.outputCount(); i++) {
                    var d = p.output(i).descriptor();
                    assertEquals(i == 0 ? roles.get(i).descriptor().shape()
                            : stacked(time, roles.get(i).descriptor().shape()), d.shape());
                    assertEquals(d.shape().isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(d.shape()))
                            : Optional.empty(), d.layout());
                    assertTrue(d.requiresGrad());
                    assertEquals(i, p.output(i).provenance().orElseThrow().outputIndex());
                    assertTrue(p.output(i).hostStorage().isEmpty());
                }
                assertFalse(h.descriptor().requiresGrad());
                assertTrue(x.descriptor().requiresGrad());
                var second = Scan.apply(List.of(initial), List.of(sequence), List.of(parameter), body, reverse);
                assertNotSame(p, second.finalCarry().getFirst().provenance().orElseThrow().producer());
            }
        }
    }

    @Test void canonicalSymbolUsesValueEqualityAndIncludesUnusedDeclarationsAndGradientFlags() {
        Tensor h = leaf(false, 2), x = leaf(false, 2);
        Shape sameName = Shape.ofDimensions(new DynamicDimension("  Rows  "), new StaticDimension(6));
        Tensor unused = source(true, sameName);
        var body = body(h, x, List.of(unused), h, h);
        var scan = Scan.apply(List.of(leaf(false, 2)), List.of(leaf(false, 0, 2)),
                List.of(source(true, row(6))), body, false);
        assertEquals(List.of(h), body.retainedValues());
        var p = scan.finalCarry().getFirst().provenance().orElseThrow().producer();
        assertTrue(p.outputDescriptors().stream().allMatch(TensorDescriptor::requiresGrad));
        assertTrue(p.outputDescriptors().stream().allMatch(d -> d.shape().isFullyStatic() && d.layout().isPresent()));
        Tensor tied = source(false, row(2));
        Tensor a = source(false, row(2)), b = source(false, row(2)), step = source(false, row(2));
        var repeated = new ScanBody(List.of(a, b), List.of(step), List.of(), List.of(a, a), List.of(a, a));
        var result = Scan.apply(List.of(tied, tied), List.of(source(false, stacked(0, row(2)))), List.of(), repeated, true);
        assertEquals(List.of(a, b), repeated.retainedValues());
        assertNotSame(result.finalCarry().get(0), result.finalCarry().get(1));
        assertNotSame(result.outputs().get(0), result.outputs().get(1));
    }

    @Test void admitsOrdinaryDtypesAndUnusedDiscreteSiblingsWithoutACpuWhitelist() throws Exception {
        Tensor h = source(DataType.FLOAT16, false, row(4)), x = source(DataType.FLOAT16, false, row(4));
        var low = body(h, x, List.of(), h.add(x).tanh());
        assertDoesNotThrow(() -> Scan.apply(List.of(source(DataType.FLOAT16, false, row(4))),
                List.of(source(DataType.FLOAT16, false, stacked(0, row(4)))), List.of(), low, false));
        Tensor f = source(false, row(4)), step = source(false, row(4));
        var top = f.relu().topK(4, 1, true, true);
        long before = nextId();
        var body = body(f, step, List.of(), top.values());
        assertEquals(before, nextId());
        assertEquals(List.of(f, top.values().provenance().orElseThrow().inputs().getFirst(),
                top.values(), top.indices()), body.retainedValues());
        var scan = Scan.apply(List.of(source(false, row(4))), List.of(source(false, stacked(3, row(4)))), List.of(), body, false);
        var indices = scan.residuals().getLast().descriptor();
        assertEquals(DataType.INT64, indices.dataType());
        assertFalse(indices.requiresGrad());
        assertEquals(stacked(3, row(4)), indices.shape());
        assertTrue(indices.layout().isEmpty());
    }

    @Test void admitsProvedCoordinateEditsTransposeContractionAndSharedReductions() {
        Tensor h = source(false, row(6)), x = source(false, row(6)), w = leaf(false, 6, 4);
        var featureEdit = Shape.ofDimensions(ROWS, new StaticDimension(2), new StaticDimension(3));
        Tensor shared = h.sum(0, false), product = h.matmul(w);
        assertDoesNotThrow(() -> body(h, x, List.of(w), h,
                h.reshape(featureEdit), h.permute(1, 0), product, shared,
                h.sumToShape(Shape.of(1, 6)), shared.expand(row(6)),
                TensorSlicePlacementExpressions.cropToShape(h, row(3), Shape.of(0, 1)),
                h.permute(1, 0).matmul(x)));
        assertEquals(Shape.of(6, 6), h.permute(1, 0).matmul(x).descriptor().shape());
    }

    @Test void scopedOrdinaryProofRejectsDeferredCountsAndForeignZeroSymbolsWithoutIds() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            var input = new TensorDescriptor(type, row(6), Optional.empty(), false);
            var wrongCount = new Operation(ShapeTransformKind.RESHAPE, new TargetShapeAttrs(row(3)));
            rejectsWithoutIds(() -> ScanBody.describeOrdinary(
                    wrongCount, List.of(input), 1, Optional.of(ROWS)));
            var sameShape = new Operation(ShapeTransformKind.RESHAPE, new TargetShapeAttrs(row(6)));
            rejectsWithoutIds(() -> ScanBody.describeOrdinary(
                    sameShape, List.of(input), 1, Optional.empty()));

            var zero = new TensorDescriptor(type, row(0), Optional.empty(), false);
            var foreign = Shape.ofDimensions(new DynamicDimension("Other"), new StaticDimension(0));
            var hiddenByZero = new Operation(ShapeTransformKind.RESHAPE, new TargetShapeAttrs(foreign));
            rejectsWithoutIds(() -> ScanBody.describeOrdinary(
                    hiddenByZero, List.of(zero), 1, Optional.of(ROWS)));
        }
    }

    @Test void rejectsAllUnprovedReshapeExpandReductionAndContractionWithoutIds() throws Exception {
        Tensor h = source(false, row(6)), x = source(false, row(6));
        var invalid = List.of(h.reshape(row(3)), h.reshape(Shape.of(6)),
                h.reshape(Shape.ofDimensions(ROWS, ROWS)),
                TensorSlicePlacementExpressions.cropToShape(h, Shape.of(32, 3), Shape.of(0, 1)),
                h.expand(Shape.of(32, 6)), h.expand(Shape.of(1, 6)), h.sumToShape(Shape.of(32, 6)),
                h.permute(1, 0).matmul(leaf(false, 32, 6)), h.select(0, 0));
        for (Tensor value : invalid) rejectsWithoutIds(() -> body(h, x, List.of(), h, value));
        Tensor weight = leaf(false, 32, 6);
        Tensor contraction = h.permute(1, 0).matmul(weight);
        rejectsWithoutIds(() -> body(h, x, List.of(weight), h, contraction));
        Tensor batched = source(false, Shape.ofDimensions(ROWS, new StaticDimension(2), new StaticDimension(6)));
        Tensor fixedBatched = leaf(false, 32, 6, 4);
        Tensor unprovedBatch = batched.matmul(fixedBatched);
        rejectsWithoutIds(() -> body(h, x, List.of(batched, fixedBatched), h, unprovedBatch));
        // Ordinary Model families already reject the non-singleton broadcast, even before Scan.
        Tensor fixed = leaf(false, 32, 6), condition = source(DataType.BOOL, false, row(6));
        rejectsWithoutIds(() -> h.add(fixed));
        rejectsWithoutIds(() -> TensorWhereExpressions.apply(condition, h, fixed));
        // A directly forged producer must still meet those same ordinary guards.
        var wrong = new TensorDescriptor(DataType.FLOAT32, Shape.of(32, 6), Optional.empty(), false);
        var binary = new TensorProducer(new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                List.of(h, fixed), List.of(wrong));
        var where = new TensorProducer(new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                List.of(condition, h, fixed), List.of(wrong));
        rejectsWithoutIds(() -> body(h, x, List.of(fixed), h, binary.output(0)));
        rejectsWithoutIds(() -> body(h, x, List.of(fixed, condition), h, where.output(0)));
    }

    @Test void rejectsIndependentRepeatedDerivedAnonymousAndOutputOnlySymbolsAcrossUnusedSiblings() throws Exception {
        Tensor h = source(false, row(4)), x = source(false, row(4));
        var invalid = List.of(
                Shape.ofDimensions(new DynamicDimension("Other"), new StaticDimension(4)),
                Shape.ofDimensions(ROWS, ROWS),
                Shape.ofDimensions(DimensionExpressions.addConstant(ROWS, 1), new StaticDimension(4)),
                Shape.ofDimensions(DimensionExpressions.unknown(0, Optional.empty()), new StaticDimension(4)));
        for (Shape shape : invalid) {
            Tensor unused = source(false, shape);
            rejectsWithoutIds(() -> body(h, x, List.of(unused), h));
        }
        Tensor fixed = leaf(false, 4), fixedStep = leaf(false, 4);
        Tensor introduced = fixed.expand(row(4));
        rejectsWithoutIds(() -> body(fixed, fixedStep, List.of(), fixed, introduced));
        var good = new TensorDescriptor(DataType.FLOAT32, row(4), Optional.empty(), false);
        var badSibling = new TensorDescriptor(DataType.INT64, invalid.getFirst(), Optional.empty(), false);
        var malformed = new TensorProducer(new Operation(TopKKind.TOP_K, new TopKAttrs(1, 4, true, true)),
                List.of(h), List.of(good, badSibling));
        rejectsWithoutIds(() -> body(h, x, List.of(), malformed.output(0)));
        var malformedType = new TensorProducer(new Operation(TopKKind.TOP_K, new TopKAttrs(1, 4, true, true)),
                List.of(h), List.of(good, good));
        rejectsWithoutIds(() -> body(h, x, List.of(), malformedType.output(0)));
    }

    @Test void attentionRequiresExactBatchContractionSequenceAndMaskShapesWithoutIds() throws Exception {
        Tensor q = source(false, namedAt(0, 0, 2, 6)), step = source(false, q.descriptor().shape());
        Tensor k = leaf(false, 32, 3, 6), v = leaf(false, 32, 3, 4);
        Tensor specialized = q.scaledDotProductAttention(k, v);
        assertEquals(Shape.of(32, 2, 4), specialized.descriptor().shape());
        rejectsWithoutIds(() -> body(q, step, List.of(k, v), q, specialized));
        Tensor emb = source(false, namedAt(2, 1, 2, 0)), embStep = source(false, emb.descriptor().shape());
        Tensor badEmbedding = emb.scaledDotProductAttention(k, v);
        rejectsWithoutIds(() -> body(emb, embStep, List.of(k, v), emb, badEmbedding));
        Tensor keySequence = source(false, namedAt(1, 1, 0, 6));
        Tensor fixedQuery = leaf(false, 1, 2, 6), fixedStep = leaf(false, 1, 2, 6);
        Tensor fixedValue = leaf(false, 1, 3, 4);
        Tensor badSequence = fixedQuery.scaledDotProductAttention(keySequence, fixedValue);
        rejectsWithoutIds(() -> body(fixedQuery, fixedStep, List.of(keySequence, fixedValue), fixedQuery, badSequence));
        Tensor fixedKey = leaf(false, 1, 3, 6);
        Tensor mask = source(DataType.BOOL, false, namedAt(1, 1, 0, 3));
        Tensor badMask = fixedQuery.scaledDotProductAttention(fixedKey, fixedValue, mask);
        rejectsWithoutIds(() -> body(fixedQuery, fixedStep, List.of(fixedKey, fixedValue, mask), fixedQuery, badMask));
        // Each applicable dtype remains source-valid, with singleton/shared batch operands too.
        for (DataType type : List.of(DataType.FLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            Tensor query = source(type, false, namedAt(0, 0, 2, 6)), xs = source(type, false, query.descriptor().shape());
            Tensor key = source(type, false, namedAt(0, 0, 3, 6));
            Tensor value = source(type, false, namedAt(0, 0, 3, 4));
            Tensor sharedKey = source(type, false, Shape.of(3, 6)), sharedValue = source(type, false, Shape.of(1, 3, 4));
            Tensor goodMask = source(DataType.BOOL, false, namedAt(0, 0, 1, 3));
            var weights = query.scaledDotProductAttentionWithWeights(key, value, goodMask);
            Tensor sharedAttention = query.scaledDotProductAttention(sharedKey, sharedValue);
            long before = nextId();
            var b = body(query, xs, List.of(key, value, sharedKey, sharedValue, goodMask), query,
                    weights.output(), sharedAttention);
            assertEquals(before, nextId());
            assertTrue(b.retainedValues().contains(weights.weights()));
            assertEquals(namedAt(0, 0, 2, 4), weights.output().descriptor().shape());
        }
    }

    @Test void normalizationRejectsHiddenAndSpecializingAttributeShapesWithoutNarrowingDtypes() throws Exception {
        Tensor h = source(false, row(6)), x = source(false, row(6));
        var epsilon = ScalarValue.float32(0.00001f);
        List<Shape> badShapes = List.of(
                Shape.ofDimensions(new DynamicDimension("Other"), new StaticDimension(6)),
                Shape.ofDimensions(DimensionExpressions.addConstant(ROWS, 1), new StaticDimension(6)),
                Shape.ofDimensions(DimensionExpressions.unknown(0, Optional.empty()), new StaticDimension(6)),
                Shape.ofDimensions(ROWS, ROWS),
                Shape.of(32, 6));
        for (Shape shape : badShapes) {
            Tensor layer = h.layerNorm(shape, epsilon), rms = h.rmsNorm(shape, epsilon);
            rejectsWithoutIds(() -> body(h, x, List.of(), h, layer));
            rejectsWithoutIds(() -> body(h, x, List.of(), h, rms));
        }
        Tensor fixed = leaf(false, 32, 6), fixedStep = leaf(false, 32, 6), unused = source(false, row(1));
        Tensor hidden = fixed.layerNorm(row(6), epsilon), hiddenRms = fixed.rmsNorm(row(6), epsilon);
        Tensor hiddenCrop = TensorSlicePlacementExpressions.cropToShape(fixed, Shape.of(32, 6),
                Shape.ofDimensions(new DynamicDimension("Other"), new StaticDimension(0)));
        rejectsWithoutIds(() -> body(fixed, fixedStep, List.of(unused), fixed, hidden));
        rejectsWithoutIds(() -> body(fixed, fixedStep, List.of(unused), fixed, hiddenRms));
        rejectsWithoutIds(() -> body(fixed, fixedStep, List.of(unused), fixed, hiddenCrop));
        Tensor affineScale = source(false, row(6)), affineBias = source(false, row(6));
        Tensor hiddenAffine = fixed.layerNorm(row(6), affineScale, affineBias, epsilon);
        Tensor hiddenAffineRms = fixed.rmsNorm(row(6), affineScale, epsilon);
        rejectsWithoutIds(() -> body(fixed, fixedStep, List.of(affineScale, affineBias), fixed, hiddenAffine));
        rejectsWithoutIds(() -> body(fixed, fixedStep, List.of(affineScale), fixed, hiddenAffineRms));
        for (DataType type : List.of(DataType.FLOAT16, DataType.FLOAT32)) {
            Tensor input = source(type, false, row(6)), step = source(type, false, row(6));
            Tensor scale = source(type, false, row(6)), bias = source(type, false, row(6));
            Tensor layer = input.layerNorm(row(6), scale, bias, epsilon);
            Tensor rms = input.rmsNorm(row(6), scale, epsilon);
            Tensor featureOnly = input.layerNorm(Shape.of(6), epsilon);
            long before = nextId();
            assertDoesNotThrow(() -> body(input, step, List.of(scale, bias), input, layer, rms, featureOnly));
            assertEquals(before, nextId());
        }
    }

    @Test void lossEqualityIncludesDenseAndIndexTargetsEvenWhenReductionMakesStaticOutput() throws Exception {
        Tensor h = source(false, row(6)), step = source(false, row(6));
        Tensor dense = leaf(false, 32, 6), indices = leaf(DataType.INT64, false, 32);
        var invalid = List.of(h.meanSquaredError(dense, LossReduction.SUM),
                h.categoricalCrossEntropyWithLogits(dense, 1, LossReduction.MEAN),
                h.categoricalCrossEntropyWithLogits(indices, 1, LossReduction.SUM));
        for (Tensor output : invalid) rejectsWithoutIds(() -> body(h, step, List.of(dense, indices), h, output));
        Tensor exact = source(false, row(6));
        Tensor exactIndices = source(DataType.INT64, false, Shape.ofDimensions(ROWS));
        var valid = List.of(h.meanSquaredError(exact, LossReduction.SUM),
                h.categoricalCrossEntropyWithLogits(exact, 1, LossReduction.MEAN),
                h.categoricalCrossEntropyWithLogits(exactIndices, 1, LossReduction.NONE));
        long before = nextId();
        assertDoesNotThrow(() -> body(h, step, List.of(exact, exactIndices), h, valid.toArray(Tensor[]::new)));
        assertEquals(before, nextId());
    }

    @Test void convolutionProvesGroupedChannelsAndBiasWithoutLosingSymbolicBatchSource() throws Exception {
        var c2 = new Conv2dAttrs(1, 1, 0, 0, 1, 1, 1);
        var c3 = new Conv3dAttrs(1, 1, 1, 0, 0, 0, 1, 1, 1, 1);
        Tensor input = source(false, namedAt(1, 1, 0, 2, 2)), step = source(false, input.descriptor().shape());
        Tensor weight = leaf(false, 4, 32, 1, 1);
        Tensor bad = input.conv2d(weight, c2);
        rejectsWithoutIds(() -> body(input, step, List.of(weight), input, bad));
        Tensor volume = source(false, namedAt(1, 1, 0, 2, 2, 2)), volumeStep = source(false, volume.descriptor().shape());
        Tensor volumeWeight = leaf(false, 4, 32, 1, 1, 1);
        Tensor bad3d = volume.conv3d(volumeWeight, c3);
        rejectsWithoutIds(() -> body(volume, volumeStep, List.of(volumeWeight), volume, bad3d));
        Tensor fixed = leaf(false, 1, 6, 2, 2), fixedStep = leaf(false, 1, 6, 2, 2);
        Tensor fixedWeight = leaf(false, 4, 6, 1, 1), bias = source(false, Shape.ofDimensions(ROWS));
        Tensor badBias = fixed.conv2d(fixedWeight, bias, c2);
        rejectsWithoutIds(() -> body(fixed, fixedStep, List.of(fixedWeight, bias), fixed, badBias));
        // A symbolic channel relation must be exact too, not just syntactically one symbol.
        Tensor channelWeight = source(false, namedAt(1, 4, 0, 1, 1));
        Tensor exact = input.conv2d(channelWeight, c2);
        Tensor grouped = input.conv2d(channelWeight, new Conv2dAttrs(1, 1, 0, 0, 1, 1, 2));
        rejectsWithoutIds(() -> body(input, step, List.of(channelWeight), input, grouped));
        Tensor rowInput = source(false, namedAt(0, 0, 6, 2, 2));
        Tensor rowStep = source(false, rowInput.descriptor().shape());
        Tensor rowOutput = rowInput.conv2d(fixedWeight, c2);
        Tensor rowVolume = source(false, namedAt(0, 0, 6, 2, 2, 2));
        Tensor rowVolumeStep = source(false, rowVolume.descriptor().shape()), vw = leaf(false, 4, 6, 1, 1, 1);
        Tensor rowVolumeOutput = rowVolume.conv3d(vw, c3);
        long before = nextId();
        assertDoesNotThrow(() -> body(input, step, List.of(channelWeight), input, exact));
        assertDoesNotThrow(() -> body(rowInput, rowStep, List.of(fixedWeight), rowInput, rowOutput));
        assertDoesNotThrow(() -> body(rowVolume, rowVolumeStep, List.of(vw), rowVolume, rowVolumeOutput));
        assertEquals(before, nextId());
    }

    @Test void batchNormalizationProvesEveryParameterChannelIncludingUnusedTrainingSiblings() throws Exception {
        Tensor h = source(false, row(6)), step = source(false, row(6));
        var epsilon = ScalarValue.float32(0.00001f);
        var momentum = ScalarValue.float32(0.1f);
        Tensor a = leaf(false, 6), b = leaf(false, 6), c = leaf(false, 6), d = leaf(false, 6);
        Tensor symbolic = source(false, Shape.ofDimensions(ROWS));
        for (int slot = 0; slot < 4; slot++) {
            var args = new ArrayList<>(List.of(a, b, c, d)); args.set(slot, symbolic);
            Tensor inference = h.batchNormInference(1, args.get(0), args.get(1), args.get(2), args.get(3), epsilon);
            Tensor training = h.batchNormTraining(1, args.get(0), args.get(1), args.get(2), args.get(3), momentum, epsilon).output();
            rejectsWithoutIds(() -> body(h, step, args, h, inference));
            rejectsWithoutIds(() -> body(h, step, args, h, training));
        }
        Tensor inference = h.batchNormInference(1, a, b, c, d, epsilon);
        var training = h.batchNormTraining(1, a, b, c, d, momentum, epsilon);
        long before = nextId();
        var valid = body(h, step, List.of(a, b, c, d), h, inference, training.output());
        assertEquals(before, nextId());
        assertTrue(valid.retainedValues().contains(training.nextRunningVariance()));
        assertEquals(5, training.output().provenance().orElseThrow().producer().outputCount());
    }

    @Test void retainedStrictGatherScatterAndFoldGuardsProveNamedShapesWithoutNewInference() throws Exception {
        Tensor h = source(false, row(6)), step = source(false, row(6));
        Tensor indices = source(DataType.INT64, false, row(2));
        Tensor tuples = source(DataType.INT64, false, namedAt(0, 0, 2, 1));
        Tensor updates = source(false, row(2));
        Tensor gathered = h.gatherElements(indices, 1), nd = h.gatherNd(tuples, 1);
        Tensor scattered = h.scatterElements(indices, updates, 1);
        Tensor scatterNd = h.scatterNd(tuples, updates, io.github.pho001.synaptik.model.operation.index.ScatterReduction.NONE, 1);
        Tensor image = source(false, namedAt(0, 0, 2, 2, 2)), imageStep = source(false, image.descriptor().shape());
        var window2 = new Window2dAttrs(1, 1, 1, 1, 0, 0, 1, 1, false);
        Tensor columns = image.unfold2d(window2), folded = columns.fold2d(image.descriptor().shape(), window2);
        Tensor volume = source(false, namedAt(0, 0, 2, 2, 2, 2)), volumeStep = source(false, volume.descriptor().shape());
        var window3 = new Window3dAttrs(1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
        Tensor volumeColumns = volume.unfold3d(window3), foldedVolume = volumeColumns.fold3d(volume.descriptor().shape(), window3);
        long before = nextId();
        assertDoesNotThrow(() -> body(h, step, List.of(indices, tuples, updates), h, gathered, nd, scattered, scatterNd));
        assertDoesNotThrow(() -> body(image, imageStep, List.of(), image, folded));
        assertDoesNotThrow(() -> body(volume, volumeStep, List.of(), volume, foldedVolume));
        assertEquals(before, nextId());
        Tensor fixedIndices = leaf(DataType.INT64, false, 32, 2);
        rejectsWithoutIds(() -> h.gatherElements(fixedIndices, 1));
        rejectsWithoutIds(() -> h.scatterElements(fixedIndices, updates, 1));
        rejectsWithoutIds(() -> columns.fold2d(Shape.of(32, 2, 2, 2), window2));
        rejectsWithoutIds(() -> volumeColumns.fold3d(Shape.of(32, 2, 2, 2, 2), window3));
    }

    @Test void actualBindingsRequireExactNamedShapesFlagsTypesDisjointnessAndStaticCommonTime() throws Exception {
        Tensor h = source(false, row(4)), x = source(true, row(4)), y = source(true, row(4));
        var body = new ScanBody(List.of(h), List.of(x, y), List.of(), List.of(h.add(x)), List.of());
        Tensor initial = source(false, row(4)), seq = source(true, stacked(3, row(4)));
        for (Tensor bad : List.of(leaf(false, 32, 4), source(true, row(4)), source(DataType.FLOAT64, false, row(4)),
                source(false, Shape.ofDimensions(new DynamicDimension("Other"), new StaticDimension(4))), h)) {
            rejectsWithoutIds(() -> Scan.apply(List.of(bad), List.of(seq, seq), List.of(), body, false));
        }
        for (Tensor bad : List.of(source(true, stacked(4, row(4))), source(true, row(4)),
                source(false, stacked(3, row(4))), source(true, Shape.of(3, 32, 4)),
                source(true, Shape.ofDimensions(ROWS, ROWS, new StaticDimension(4))))) {
            rejectsWithoutIds(() -> Scan.apply(List.of(initial), List.of(seq, bad), List.of(), body, false));
        }
        Tensor fh = leaf(false, 32, 4), fx = leaf(true, 32, 4);
        var fixed = body(fh, fx, List.of(), fh);
        Tensor fixedSequence = leaf(true, 3, 32, 4);
        rejectsWithoutIds(() -> Scan.apply(List.of(initial), List.of(fixedSequence), List.of(), fixed, false));
    }

    @Test void directAttachmentsCheckEveryPortIncludingLastResidualBeforeOutputSupplier() throws Exception {
        Tensor h = source(false, row(4)), x = source(false, row(4));
        var body = body(h, x, List.of(), h.add(x).relu(), h);
        var scan = Scan.apply(List.of(source(false, row(4))), List.of(source(false, stacked(0, row(4)))), List.of(), body, false);
        var p = scan.finalCarry().getFirst().provenance().orElseThrow().producer();
        for (int index = 0; index < p.outputCount(); index++) {
            var outputs = new ArrayList<>(p.outputDescriptors());
            var original = outputs.get(index);
            outputs.set(index, new TensorDescriptor(original.dataType(), index == 0 ? row(3) : stacked(0, row(3)), Optional.empty(), false));
            var count = new AtomicInteger();
            rejectsWithoutIds(() -> new TensorProducer(p.operation(), p.inputs(), outputs, Optional.empty(),
                    () -> new TensorId(count.getAndIncrement()), Optional.of(body)));
            assertEquals(0, count.get());
        }
        var wrongTime = new Operation(ScanKind.SCAN, new ScanAttrs(1, 1, 1, 0, 1, body.retainedValues().size(), false));
        var count = new AtomicInteger();
        rejectsWithoutIds(() -> new TensorProducer(wrongTime, p.inputs(), p.outputDescriptors(), Optional.empty(),
                () -> new TensorId(count.getAndIncrement()), Optional.of(body)));
        assertEquals(0, count.get());
        Tensor shared = leaf(false, 4);
        var mixed = body(h, x, List.of(shared), h, shared);
        var mixedScan = Scan.apply(p.inputs().subList(0, 1), p.inputs().subList(1, 2), List.of(leaf(false, 4)), mixed, false);
        var mp = mixedScan.finalCarry().getFirst().provenance().orElseThrow().producer();
        var missingStaticLayout = new ArrayList<>(mp.outputDescriptors());
        missingStaticLayout.set(1, new TensorDescriptor(DataType.FLOAT32, Shape.of(0, 4), Optional.empty(), false));
        rejectsWithoutIds(() -> new TensorProducer(mp.operation(), mp.inputs(), missingStaticLayout, Optional.empty(),
                () -> new TensorId(count.getAndIncrement()), Optional.of(mixed)));
        var wrongFlag = new ArrayList<>(p.outputDescriptors());
        var last = wrongFlag.getLast();
        wrongFlag.set(wrongFlag.size() - 1, new TensorDescriptor(last.dataType(), last.shape(), Optional.empty(), true));
        rejectsWithoutIds(() -> new TensorProducer(p.operation(), p.inputs(), wrongFlag, Optional.empty(),
                () -> new TensorId(count.getAndIncrement()), Optional.of(body)));
        assertEquals(0, count.get());
    }

    @Test void lengthApplyAndDirectAttachmentRequireExactBatchAndStaticSharedShapesBeforeIds() throws Exception {
        Tensor h = source(false, row(4)), x = source(false, row(4));
        var body = body(h, x, List.of(), h, h);
        Tensor initial = source(false, row(4)), seq = source(false, stacked(0, row(4))), lengths = leaf(DataType.INT64, false, 32);
        var allValid = Scan.apply(List.of(initial), List.of(seq), List.of(), body, false);
        var p = allValid.finalCarry().getFirst().provenance().orElseThrow().producer();
        rejectsWithoutIds(() -> Scan.apply(List.of(initial), List.of(seq), List.of(), lengths, 1,
                List.of(OptionalInt.of(0)), body, false));
        var operation = new Operation(ScanKind.SCAN, new ScanAttrs(0, 1, 1, 0, 1, 1, false, true, 1, List.of(OptionalInt.of(0))));
        var count = new AtomicInteger();
        var failure = assertThrows(IllegalArgumentException.class, () -> new TensorProducer(operation,
                List.of(initial, seq, lengths), p.outputDescriptors(), Optional.empty(),
                () -> new TensorId(count.getAndIncrement()), Optional.of(body)));
        assertTrue(failure.getMessage().contains("length-aware scan row semantics"));
        assertEquals(0, count.get());
        // An unused SHARED declaration cannot contain a named batch, even with static outputs.
        Tensor fixed = leaf(false, 32, 4), step = leaf(false, 32, 4), unused = source(false, row(4));
        var hidden = body(fixed, step, List.of(unused), fixed);
        Tensor actual = leaf(false, 32, 4), sequence = leaf(false, 0, 32, 4), invariant = source(false, row(4));
        rejectsWithoutIds(() -> Scan.apply(List.of(actual), List.of(sequence), List.of(invariant), lengths,
                1, List.of(OptionalInt.of(0)), hidden, false));
        var hiddenSource = Scan.apply(List.of(actual), List.of(sequence), List.of(invariant), hidden, false);
        var hp = hiddenSource.finalCarry().getFirst().provenance().orElseThrow().producer();
        var hiddenOperation = new Operation(ScanKind.SCAN, new ScanAttrs(0, 1, 1, 1, 0, 1, false, true, 1, List.of(OptionalInt.of(0))));
        var hiddenFailure = assertThrows(IllegalArgumentException.class, () -> new TensorProducer(hiddenOperation,
                List.of(actual, sequence, invariant, lengths), hp.outputDescriptors(), Optional.empty(),
                () -> new TensorId(count.getAndIncrement()), Optional.of(hidden)));
        assertTrue(hiddenFailure.getMessage().contains("length-aware scan row semantics"));
        assertEquals(0, count.get());
        Tensor symbolicLengths = source(DataType.INT64, false, Shape.ofDimensions(ROWS));
        var staticBody = body(fixed, step, List.of(), fixed);
        rejectsWithoutIds(() -> Scan.apply(List.of(actual), List.of(sequence), List.of(), symbolicLengths,
                1, List.of(OptionalInt.of(0)), staticBody, false));
    }

    @Test void storageFreeContainmentStillRejectsFreeEffectfulNestedAndAliasValues() throws Exception {
        Tensor h = source(false, row(4)), x = source(false, row(4)), free = source(false, row(4));
        Tensor bad = h.add(free);
        rejectsWithoutIds(() -> body(h, x, List.of(), bad));
        Tensor next = h.tanh();
        var alias = new Tensor(new TensorId(900001), next.descriptor(), Optional.empty(), next.provenance(), Optional.empty());
        rejectsWithoutIds(() -> body(h, x, List.of(), alias));
        var dropout = h.dropout(0.5, GraphRngState.initial(1, 0));
        rejectsWithoutIds(() -> body(h, x, List.of(), dropout.output()));
        Tensor innerH = source(false, row(4)), innerX = source(false, row(4));
        var nestedBody = body(innerH, innerX, List.of(), innerH);
        Tensor nested = Scan.apply(List.of(h), List.of(x.expandDims(0)), List.of(), nestedBody, false).finalCarry().getFirst();
        rejectsWithoutIds(() -> body(h, x, List.of(), nested));
        // Static contained values may still acquire storage after construction.
        Tensor invariant = leaf(false, 4), actualInvariant = leaf(false, 4);
        var b = body(h, x, List.of(invariant), h.add(invariant));
        Tensor initial = source(false, row(4)), sequence = source(false, stacked(3, row(4)));
        var valid = Scan.apply(List.of(initial), List.of(sequence), List.of(actualInvariant), b, false);
        var p = valid.finalCarry().getFirst().provenance().orElseThrow().producer();
        invariant.replaceHostStorage(TensorFactory.zeros(Shape.of(4), DataType.FLOAT32, Optional.empty(), false)
                .hostStorage().orElseThrow());
        rejectsWithoutIds(() -> Scan.apply(List.of(initial), List.of(sequence), List.of(actualInvariant), b, false));
        var count = new AtomicInteger();
        rejectsWithoutIds(() -> new TensorProducer(p.operation(), p.inputs(), p.outputDescriptors(), Optional.empty(),
                () -> new TensorId(count.getAndIncrement()), Optional.of(b)));
        assertEquals(0, count.get());
    }
}

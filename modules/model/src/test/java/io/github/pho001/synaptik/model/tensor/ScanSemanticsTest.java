package io.github.pho001.synaptik.model.tensor;

import static io.github.pho001.synaptik.model.tensor.ScanBodyTest.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalInt;
import java.util.Set;
import java.util.Collections;
import java.util.IdentityHashMap;
import org.junit.jupiter.api.Test;

class ScanSemanticsTest {
    @Test
    void namedFeatureOverflowChecksUnusedDeclarationsAndContainedValuesBeforeOutputIds() throws Exception {
        var batch = new io.github.pho001.synaptik.model.shape.DynamicDimension("Customers");
        Shape hugeShape = ScanSymbolicLengthSourceTest.shape(batch,
                ScanSymbolicLengthSourceTest.fixed(Long.MAX_VALUE), ScanSymbolicLengthSourceTest.fixed(2));
        Tensor h = ScanSymbolicLengthSourceTest.source(DataType.FLOAT32, false,
                ScanSymbolicLengthSourceTest.shape(batch, ScanSymbolicLengthSourceTest.fixed(1), ScanSymbolicLengthSourceTest.fixed(1)));
        Tensor x = ScanSymbolicLengthSourceTest.fresh(h, h.descriptor().shape());
        Tensor huge = ScanSymbolicLengthSourceTest.fresh(h, hugeShape);
        var bodies = List.of(
                new ScanBody(List.of(huge), List.of(x), List.of(), List.of(huge), List.of()),
                new ScanBody(List.of(h), List.of(x, huge), List.of(), List.of(h), List.of()),
                body(h, x, List.of(), h, h.expand(hugeShape)));
        for (ScanBody body : bodies) for (long time : new long[] {0, 3}) {
            var carry = body.carry().stream().map(t -> ScanSymbolicLengthSourceTest.fresh(t, t.descriptor().shape())).toList();
            var seq = body.step().stream().map(t -> ScanSymbolicLengthSourceTest.fresh(t,
                    ScanSymbolicLengthSourceTest.stacked(time, t.descriptor().shape()))).toList();
            Tensor lengths = ScanSymbolicLengthSourceTest.source(DataType.INT64, false,
                    io.github.pho001.synaptik.model.shape.Shape.ofDimensions(batch));
            // Broader all-valid source must not acquire the new length-only admission rule.
            var allValid = Scan.apply(carry, seq, List.of(), body, false);
            var p = allValid.finalCarry().getFirst().provenance().orElseThrow().producer();
            var axes = body.step().stream().map(t -> OptionalInt.of(0)).toList();
            var attrs = new io.github.pho001.synaptik.model.operation.scan.ScanAttrs(time, 1, seq.size(), 0, body.emissions().size(), body.retainedValues().size(), false, true, 1, axes);
            var inputs = new ArrayList<>(p.inputs()); inputs.add(lengths);
            var supplied = new java.util.concurrent.atomic.AtomicInteger(); long before = nextId();
            assertThrows(ArithmeticException.class, () -> Scan.apply(carry, seq, List.of(), lengths, 1, axes, body, false));
            assertThrows(ArithmeticException.class, () -> new TensorProducer(new io.github.pho001.synaptik.model.operation.Operation(
                    io.github.pho001.synaptik.model.operation.scan.ScanKind.SCAN, attrs), inputs, p.outputDescriptors(), java.util.Optional.empty(),
                    () -> new TensorId(supplied.getAndIncrement()), java.util.Optional.of(body)));
            assertEquals(before, nextId()); assertEquals(0, supplied.get());
        }
        Shape zeroFeatures = ScanSymbolicLengthSourceTest.shape(batch, ScanSymbolicLengthSourceTest.fixed(Long.MAX_VALUE),
                ScanSymbolicLengthSourceTest.fixed(2), ScanSymbolicLengthSourceTest.fixed(0));
        Tensor empty = ScanSymbolicLengthSourceTest.fresh(h, zeroFeatures);
        var valid = body(empty, x, List.of(), empty);
        assertDoesNotThrow(() -> ScanSymbolicLengthSourceTest.apply(valid, 0, 1, List.of(OptionalInt.of(0)), false));
    }

    @Test
    void zeroFeatureProductDoesNotOverflowBeforeItsLaterZero() throws Exception {
        Tensor h = leaf(false, 1, Long.MAX_VALUE, 2, 0), x = leaf(false, 1, 0);
        Tensor edited = h.reshape(Shape.of(1, 0)).reshape(h.descriptor().shape());
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(edited), List.of());
        Tensor initial = leaf(false, 1, Long.MAX_VALUE, 2, 0), sequence = leaf(false, 0, 1, 0);
        Tensor lengths = leaf(DataType.INT64, false, 1);
        long before = nextId();
        var result = Scan.apply(List.of(initial), List.of(sequence), List.of(), lengths, 1,
                List.of(OptionalInt.of(0)), body, false);
        assertEquals(before + 1 + body.retainedValues().size(), nextId());
        assertEquals(h.descriptor().shape(), result.finalCarry().getFirst().descriptor().shape());
    }

    @Test
    void zeroBatchDoesNotExcuseOverflowingCanonicalFeatureGeometry() throws Exception {
        long before = nextId();
        // This shape cannot become a canonical formal: ordinary stride arithmetic already rejects
        // its nonzero trailing-feature overflow, despite the zero leading batch extent.
        assertThrows(ArithmeticException.class, () -> leaf(false, 0, Long.MAX_VALUE, 2));
        assertEquals(before, nextId());
    }
    @Test
    void lengthModePreservesCanonicalRolesAndSharedGeometryAcrossEmptyEqualAndSingletonDomains() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            for (long batch : new long[] {0, 1, 2}) for (long time : new long[] {0, 3}) {
                Tensor h = leaf(type, false, batch, 2), old = leaf(type, false, 2);
                Tensor x = leaf(type, true, batch, 2), transposed = leaf(type, false, 2, batch);
                Tensor sharedStep = leaf(type, false, 2, 2), bias = leaf(type, false, 1, 2);
                Tensor row = h.add(x).add(transposed.transpose()).add(bias).tanh();
                Tensor contribution = row.sum(0, false).neg().neg();
                var body = new ScanBody(List.of(h, old), List.of(x, transposed, sharedStep), List.of(bias),
                        List.of(row, old.add(contribution)), List.of(h, row, row));
                for (boolean reverse : List.of(false, true)) {
                    var result = length(body, time, batch, 1,
                            List.of(OptionalInt.of(0), OptionalInt.of(1), OptionalInt.empty()), reverse);
                    var p = result.finalCarry().getFirst().provenance().orElseThrow().producer();
                    assertSame(body, p.scanBody().orElseThrow());
                    assertEquals(7, p.inputs().size());
                    assertEquals(type, result.finalCarry().getLast().descriptor().dataType());
                    assertTrue(result.finalCarry().getLast().descriptor().requiresGrad());
                    int sharedHistory = body.retainedValues().indexOf(old);
                    assertEquals(Shape.of(time, 2), result.residuals().get(sharedHistory).descriptor().shape());
                    assertEquals(Shape.of(time, batch, 2), result.outputs().getFirst().descriptor().shape());
                    assertNotSame(result.outputs().get(1), result.outputs().get(2));
                    assertEquals(reverse, ((io.github.pho001.synaptik.model.operation.scan.ScanAttrs) p.operation().attrs()).reverse());
                }
            }
        }
    }

    @Test
    void singletonBroadcastAndSumToShapeIdentityPreserveExplicitRowIncludingRowOne() {
        Tensor h = leaf(false, 1, 2), x = leaf(false, 2, 1), bias = leaf(false, 1, 2);
        Tensor next = h.sumToShape(Shape.of(1, 2)).add(x.sumToShape(Shape.of(2, 1)).transpose()).add(bias);
        var body = new ScanBody(List.of(h), List.of(x), List.of(bias), List.of(next), List.of(next));
        assertDoesNotThrow(() -> length(body, 0, 1, 1, List.of(OptionalInt.of(1)), false));
    }

    @Test
    void featureAlgebraAndTerminalMatmulContractionRetainProvedRoles() {
        for (long batch : new long[] {0, 1, 2}) {
            Tensor h = leaf(false, batch, 2), old = leaf(false, 2, 2), x = leaf(false, batch, 2), w = leaf(false, 2, 2);
            Tensor row = h.matmul(w).add(x).reshape(Shape.of(batch, 1, 2)).expand(Shape.of(batch, 3, 2))
                    .sum(new int[] {-2}, false).expandDims(-1).squeeze(-1).contiguous().cast(DataType.FLOAT32);
            Tensor contraction = x.transpose().matmul(row).add(w).reshape(Shape.of(2, 2));
            var body = new ScanBody(List.of(h, old), List.of(x), List.of(w), List.of(row, old.add(contraction)), List.of(row));
            assertDoesNotThrow(() -> length(body, 2, batch, 1, List.of(OptionalInt.of(0)), true));
        }
    }

    @Test
    void rejectsUnprovedBroadcastShapeRowMixingAndTerminalFeedbackBeforeIds() throws Exception {
        for (long batch : new long[] {0, 1, 2}) {
            Tensor h = leaf(false, batch, 2), x = leaf(false, batch, 2), shared = leaf(false, batch, 2);
            var invalid = new ArrayList<Tensor>();
            invalid.add(shared); // equal shape cannot invent ROW
            invalid.add(h.sum(0, true).expand(Shape.of(batch, 2)).add(h));
            invalid.add(h.sum(0, false).neg().expand(Shape.of(batch, 2)).add(h));
            invalid.add(h.reshape(Shape.of(batch * 2)).reshape(Shape.of(batch, 2)));
            invalid.add(h.transpose().reshape(h.descriptor().shape())); // Equal counts cannot move the row boundary.
            invalid.add(h.sum(0, true).cast(DataType.FLOAT64).cast(DataType.FLOAT32)
                    .expand(h.descriptor().shape()).add(h));
            if (batch != 1) invalid.add(h.add(shared));
            for (Tensor bad : invalid) {
                var body = new ScanBody(List.of(h), List.of(x), List.of(shared), List.of(bad), List.of());
                rejectsLengthWithoutIds(body, 0, batch, 1, List.of(OptionalInt.of(0)));
            }
        }
        // Row contraction patterns other than ROW1 @ ROW0 must not guess based on equal extents.
        Tensor h = leaf(false, 2, 2), x = leaf(false, 2, 2);
        rejectsLengthWithoutIds(new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.matmul(x)), List.of()),
                0, 2, 1, List.of(OptionalInt.of(0)));
    }

    @Test
    void sharedCarryCannotCoupleBroadcastContributionOrLaunderItsOwnUpdatedState() throws Exception {
        Tensor h = leaf(false, 1, 2), old = leaf(false, 2), other = leaf(false, 2), x = leaf(false, 1, 2);
        Tensor contribution = h.sum(0, false), updated = old.add(contribution);
        List<ScanBody> invalid = List.of(
                new ScanBody(List.of(h, old), List.of(x), List.of(), List.of(h.add(updated), updated), List.of()),
                new ScanBody(List.of(h, old), List.of(x), List.of(), List.of(h, contribution.add(old)), List.of()),
                new ScanBody(List.of(h, old), List.of(x), List.of(), List.of(h, old.add(h.sum())), List.of()),
                new ScanBody(List.of(h, old, other), List.of(x), List.of(), List.of(h, updated, other.add(updated.neg())), List.of()),
                new ScanBody(List.of(h, old), List.of(x), List.of(), List.of(h, updated), List.of(updated.expand(Shape.of(1, 2)))));
        for (ScanBody body : invalid) rejectsLengthWithoutIds(body, 0, 1, 1, List.of(OptionalInt.of(0)));
        for (Tensor laundered : List.of(old.cast(DataType.FLOAT64).cast(DataType.FLOAT32),
                updated.cast(DataType.FLOAT64).cast(DataType.FLOAT32))) {
            rejectsLengthWithoutIds(new ScanBody(List.of(h, old), List.of(x), List.of(),
                    List.of(h.add(laundered), updated), List.of()), 0, 1, 1, List.of(OptionalInt.of(0)));
        }
        // Explicit row SUM at B=1 is terminal even if its keep-dimensions shape is unchanged.
        rejectsLengthWithoutIds(new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.sum(0, true)), List.of()),
                0, 1, 1, List.of(OptionalInt.of(0)));
    }

    @Test
    void validatesUnusedRolesButAdmitsHeterogeneousDeclarationsEvenAtZeroTime() throws Exception {
        Tensor h = leaf(false, 2, 2), x = leaf(false, 2, 2), unused = leaf(false, 2, 3, 2);
        var body = new ScanBody(List.of(h), List.of(x, unused), List.of(), List.of(h), List.of());
        rejectsLengthWithoutIds(body, 0, 2, 1, List.of(OptionalInt.of(0), OptionalInt.of(1)));
        Tensor low = leaf(DataType.FLOAT16, false, 2, 2);
        assertDoesNotThrow(() -> length(new ScanBody(List.of(h), List.of(x), List.of(low), List.of(h), List.of()),
                0, 2, 1, List.of(OptionalInt.of(0)), false));
    }

    @Test
    void sharedContributionMustMatchOwnDtypeEvenWhenOrdinaryPromotionPreservesNextCarry() throws Exception {
        for (long batch : new long[] {0, 1, 2}) {
            Tensor h = leaf(false, batch, 2), x = leaf(false, batch, 2), old = leaf(DataType.FLOAT64, false, 2);
            Tensor contribution = h.sum(0, false);
            Tensor promoted = old.add(contribution);
            assertEquals(DataType.FLOAT64, promoted.descriptor().dataType());
            var bad = new ScanBody(List.of(h, old), List.of(x), List.of(), List.of(h, promoted), List.of());
            rejectsLengthWithoutIds(bad, 0, batch, 1, List.of(OptionalInt.of(0)));
            var good = new ScanBody(List.of(h, old), List.of(x), List.of(),
                    List.of(h, old.add(contribution.cast(DataType.FLOAT64))), List.of(h));
            assertDoesNotThrow(() -> length(good, 0, batch, 1, List.of(OptionalInt.of(0)), true));
            var identity = new ScanBody(List.of(h, old), List.of(x), List.of(), List.of(h, old), List.of());
            rejectsLengthWithoutIds(identity, 0, batch, 1, List.of(OptionalInt.of(0)));
        }
    }

    private static ScanResult length(ScanBody body, long time, long batch, int rows,
            List<OptionalInt> axes, boolean reverse) {
        var carry = body.carry().stream().map(ScanSemanticsTest::actual).toList();
        var seq = body.step().stream().map(t -> {
            long[] dims = new long[t.descriptor().shape().rank() + 1];
            dims[0] = time;
            System.arraycopy(t.descriptor().shape().toLongArray(), 0, dims, 1, dims.length - 1);
            return leaf(t.descriptor().dataType(), t.descriptor().requiresGrad(), dims);
        }).toList();
        return Scan.apply(carry, seq, body.invariants().stream().map(ScanSemanticsTest::actual).toList(),
                leaf(DataType.INT64, false, batch), rows, axes, body, reverse);
    }

    private static Tensor actual(Tensor t) {
        return leaf(t.descriptor().dataType(), t.descriptor().requiresGrad(), t.descriptor().shape().toLongArray());
    }

    private static void rejectsLengthWithoutIds(ScanBody body, long time, long batch, int rows,
            List<OptionalInt> axes) throws Exception {
        var carry = body.carry().stream().map(ScanSemanticsTest::actual).toList();
        var seq = body.step().stream().map(t -> {
            long[] dims = new long[t.descriptor().shape().rank() + 1]; dims[0] = time;
            System.arraycopy(t.descriptor().shape().toLongArray(), 0, dims, 1, dims.length - 1);
            return leaf(t.descriptor().dataType(), t.descriptor().requiresGrad(), dims);
        }).toList();
        var inv = body.invariants().stream().map(ScanSemanticsTest::actual).toList();
        Tensor lengths = leaf(DataType.INT64, false, batch);
        long before = nextId();
        assertThrows(IllegalArgumentException.class, () -> Scan.apply(carry, seq, inv, lengths, rows, axes, body, false));
        assertEquals(before, nextId());
    }
    @Test
    void admitsAllModelFloatingDtypesAndReluOutsideInitialCpuDomain() throws Exception {
        for (DataType type : List.of(DataType.FLOAT16, DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            Tensor h = leaf(type, true, 2), x = leaf(type, false, 2);
            Tensor next = h.add(x).tanh();
            Tensor emission = next.relu();
            long before = nextId();
            ScanBody b = body(h, x, List.of(), next, emission);
            assertEquals(before, nextId());
            ScanResult r = Scan.apply(List.of(leaf(type, true, 2)), List.of(leaf(type, false, 0, 2)), List.of(), b, false);
            assertEquals(type, r.finalCarry().getFirst().descriptor().dataType());
            assertTrue(r.outputs().getFirst().descriptor().requiresGrad());
        }
    }

    @Test
    void appliesActualInputOrToAllDifferentiablePortsIncludingUnusedRolesAndZeroTime() {
        for (int mask = 0; mask < 8; mask++) {
            boolean hg = (mask & 1) != 0, xg = (mask & 2) != 0, pg = (mask & 4) != 0;
            for (long time : new long[] {0, 3}) {
                Tensor h = leaf(hg, 2), x = leaf(xg, 2), p = leaf(pg, 2);
                ScanBody b = body(h, x, List.of(p), h, h, h.isNaN());
                ScanResult r = Scan.apply(List.of(leaf(hg, 2)), List.of(leaf(xg, time, 2)),
                        List.of(leaf(pg, 2)), b, false);
                boolean expected = mask != 0;
                assertEquals(expected, r.finalCarry().getFirst().descriptor().requiresGrad());
                assertEquals(expected, r.outputs().getFirst().descriptor().requiresGrad());
                assertFalse(r.outputs().get(1).descriptor().requiresGrad());
                for (Tensor value : r.residuals()) {
                    assertEquals(expected && value.descriptor().dataType().isDifferentiable(),
                            value.descriptor().requiresGrad());
                }
                assertEquals(hg, h.descriptor().requiresGrad());
                assertEquals(xg, x.descriptor().requiresGrad());
                assertEquals(pg, p.descriptor().requiresGrad());
            }
        }
    }

    @Test
    void preservesTemporalEmissionEligibilityWhenCarryFormalIsNotEligible() {
        Tensor h = leaf(false, 2), x = leaf(true, 2);
        ScanBody b = body(h, x, List.of(), h.add(x), h);
        ScanResult r = Scan.apply(List.of(leaf(false, 2)), List.of(leaf(true, 4, 2)), List.of(), b, false);
        assertFalse(b.emissions().getFirst().descriptor().requiresGrad());
        assertTrue(r.finalCarry().getFirst().descriptor().requiresGrad());
        assertTrue(r.outputs().getFirst().descriptor().requiresGrad());
        assertTrue(r.residuals().stream().allMatch(t -> t.descriptor().requiresGrad()));
    }

    @Test
    void nondifferentiableCarriesAndResidualsNeverAcquireGradientFlags() {
        Tensor h = leaf(DataType.INT32, false, 2), x = leaf(DataType.INT32, false, 2), p = leaf(true);
        Tensor next = h.add(x);
        ScanBody b = body(h, x, List.of(p), next, next, next.equalTo(h));
        ScanResult r = Scan.apply(List.of(leaf(DataType.INT32, false, 2)),
                List.of(leaf(DataType.INT32, false, 3, 2)), List.of(leaf(true)), b, true);
        assertFalse(r.finalCarry().getFirst().descriptor().requiresGrad());
        assertTrue(r.residuals().stream().noneMatch(t -> t.descriptor().requiresGrad()));
        assertTrue(r.outputs().stream().noneMatch(t -> t.descriptor().requiresGrad()));
    }

    @Test
    void reusesOrdinaryGuardsAcrossKnownPureAlgebraWithoutValidationIds() throws Exception {
        Tensor h = leaf(true, 2, 2), x = leaf(false, 2, 2);
        List<Tensor> emissions = new ArrayList<>(List.of(h.add(x), h.sub(x), h.mul(x), h.div(x),
                h.neg(), h.relu(), h.cast(DataType.FLOAT16), h.matmul(x), h.transpose(),
                h.reshape(4), h.expandDims(0).squeeze(0), h.contiguous(), h.sum(),
                h.sum(0, true), h.sumToShape(Shape.of(1, 2)), h.mean(new int[] {0}, false),
                h.logSumExp(new int[] {0}, false), h.variance(new int[] {0}, false, 0),
                h.softmax(1), h.cumSum(0), h.sort(1, true), h.argsort(1, false),
                h.topK(1, 1, true, false).indices(), h.sliceAxis(0, 0, 1),
                h.sliceUpdate(x, new long[] {0}, new int[] {0}, new long[] {1}),
                h.pad(new long[] {0, 1}, new long[] {1, 0}, ScalarValue.float32(0.0f)),
                h.tile(1, 2), h.unfold(1, 1, 1), h.layerNorm(Shape.of(2), ScalarValue.float32(0.01f)),
                h.rmsNorm(Shape.of(2), ScalarValue.float32(0.01f)),
                h.meanSquaredError(x, io.github.pho001.synaptik.model.operation.loss.LossReduction.MEAN)));
        emissions.addAll(List.of(h.isFinite(), h.greaterThan(x), h.sum(1, h.isFinite()),
                h.argMax(1), h.cumProd(1), h.logSoftmax(1),
                h.cropToShape(Shape.of(1, 2), Shape.of(0, 0)), h.select(0, 0),
                h.unfold(1, 1, 1).foldAxis(1, 2, 1)));
        long before = nextId();
        ScanBody b = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x)), emissions);
        // Only the explicitly constructed next root above allocates an ID; validation allocates none.
        assertEquals(before + 1, nextId());
        assertEquals(emissions, b.emissions());
        assertAllSharedSemantics(b);
        assertEquals(before + 1, nextId());
    }

    @Test
    void admitsPureIndexingSelectionNormalizationAndAttentionWithAllSiblingMetadata() throws Exception {
        Tensor h = leaf(true, 2, 2), x = leaf(false, 2, 2);
        Tensor mask = leaf(DataType.BOOL, false, 2, 2);
        Tensor indices = leaf(DataType.INT64, false, 2);
        Tensor aligned = leaf(DataType.INT32, false, 2, 2);
        Tensor nd = leaf(DataType.INT64, false, 2, 1);
        Tensor scale = leaf(false, 2), bias = leaf(false, 2), mean = leaf(false, 2), variance = leaf(false, 2);
        ScalarValue eps = ScalarValue.float32(0.01f);
        var attention = h.scaledDotProductAttentionWithWeights(x, h, mask);
        var bn = h.batchNormTraining(1, scale, bias, mean, variance, ScalarValue.float32(0.1f), eps);
        List<Tensor> outputs = List.of(Tensor.where(mask, h, x), mask.logicalNot(), mask.logicalAnd(mask),
                h.gather(indices, 1), h.gatherElements(aligned, 0), h.gatherNd(nd),
                h.scatterAdd(indices, x, 0), h.scatterElements(aligned, x, 1), h.scatterNd(nd, x),
                indices.oneHot(3), h.add(ScalarValue.float32(1)),
                h.layerNorm(Shape.of(2), scale, bias, eps), h.rmsNorm(Shape.of(2), scale, eps),
                h.batchNormInference(1, scale, bias, mean, variance, eps), bn.output(),
                h.scaledDotProductAttention(x, h), attention.output(),
                TensorCompositionExpressions.concat(0, new Tensor[] {h, x}),
                TensorCompositionExpressions.stack(0, new Tensor[] {h, x}));
        long before = nextId();
        ScanBody b = new ScanBody(List.of(h), List.of(x),
                List.of(mask, indices, aligned, nd, scale, bias, mean, variance), List.of(h), outputs);
        assertEquals(before, nextId());
        assertTrue(b.retainedValues().contains(attention.weights()));
        TensorProducer stats = bn.output().provenance().orElseThrow().producer();
        assertTrue(b.retainedValues().contains(stats.output(3)));
        assertTrue(b.retainedValues().contains(stats.output(4)));
        assertAllSharedSemantics(b);
        assertEquals(before, nextId());
    }

    @Test
    void admitsKnownPureConvolutionPoolingAndWindowFamiliesWithoutCpuGating() throws Exception {
        Tensor h = leaf(false, 2), x = leaf(false, 2);
        Tensor image = leaf(true, 1, 1, 2, 2), filter = leaf(false, 1, 1, 1, 1), bias = leaf(false, 1);
        Tensor volume = leaf(true, 1, 1, 2, 2, 2), filter3 = leaf(false, 1, 1, 1, 1, 1);
        var window = new io.github.pho001.synaptik.model.operation.layout.Window2dAttrs(1, 1, 1, 1, 0, 0, 1, 1, false);
        var window3 = new io.github.pho001.synaptik.model.operation.layout.Window3dAttrs(1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
        Tensor columns = image.unfold2d(window), columns3 = volume.unfold3d(window3);
        List<Tensor> emissions = List.of(
                image.conv2d(filter, io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs.defaults()),
                image.conv2d(filter, bias, io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs.defaults()),
                volume.conv3d(filter3, io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs.defaults()),
                volume.conv3d(filter3, bias, io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs.defaults()),
                image.maxPool2d(new io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs(1, 1, 1, 1, 0, 0, 1, 1, false)),
                image.averagePool2d(new io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs(1, 1, 1, 1, 0, 0, 1, 1, false)),
                volume.maxPool3d(new io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs(1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, false)),
                volume.averagePool3d(new io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs(1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, false)),
                columns.fold2d(image.descriptor().shape(), window),
                columns3.fold3d(volume.descriptor().shape(), window3));
        long before = nextId();
        ScanBody body = new ScanBody(List.of(h), List.of(x), List.of(image, filter, bias, volume, filter3), List.of(h), emissions);
        assertAllSharedSemantics(body);
        assertEquals(before, nextId());
    }

    /** Exercises genuine ordinary occurrences and every canonical sibling without allocating IDs. */
    private static void assertAllSharedSemantics(ScanBody body) {
        var batch = new io.github.pho001.synaptik.model.shape.StaticDimension(7);
        Set<TensorProducer> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Tensor tensor : body.retainedValues()) tensor.provenance().ifPresent(provenance -> {
            TensorProducer producer = provenance.producer();
            if (!seen.add(producer)) return;
            var inputs = producer.inputs().stream().map(Tensor::descriptor).toList();
            var facts = inputs.stream().map(d -> Scan.declaredSemantics(d, OptionalInt.empty(), batch)).toList();
            var results = Scan.ordinarySemantics(producer.operation(), inputs, facts, producer.outputDescriptors(), batch);
            assertEquals(producer.outputCount(), results.size());
            for (var fact : results) {
                assertTrue(fact.rowAxis().isEmpty());
                assertFalse(fact.terminal());
            }
        });
    }
}

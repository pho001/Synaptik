package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.pooling.*;
import io.github.pho001.synaptik.model.shape.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Genuine ordinary producer fixtures for descriptor-only coordinate and per-output routes. */
final class ScanValueSemanticsTest {
    private static final ScalarValue EPS = ScalarValue.float64(0.01);

    @Test void featureFamiliesPreserveExactStaticAndNamedRowsWithDiscreteSiblings() {
        for (Dimension batch : batches()) {
            Tensor x = tensor(DataType.FLOAT64, batch, fixed(4));
            Tensor y = tensor(DataType.FLOAT64, batch, fixed(4));
            var facts = declarations(batch, x, 0, y, 0);
            Tensor condition = x.greaterThan(y);
            List<Tensor> outputs = List.of(x.relu(), x.pow(y), x.add(ScalarValue.float64(1)),
                    x.isFinite(), condition.logicalNot(), Tensor.where(condition, x, y),
                    x.cast(DataType.INT32), x.softmax(1), x.logSoftmax(1), x.cumSum(1), x.cumProd(1),
                    x.mean(1), x.variance(new int[] {1}, false, 0), x.l2Norm(1), x.argMax(1),
                    condition.all(1), x.sum(1, condition), x.sort(1), x.argsort(1),
                    x.topK(2, 1).indices(), x.layerNorm(Shape.of(4), EPS), x.rmsNorm(Shape.of(4), EPS),
                    x.sliceAxis(1, 1, 3), x.pad(new long[] {0, 1}, new long[] {0, 1}, ScalarValue.float64(0)),
                    x.tile(1, 2), x.unfold(1, 2, 1), x.select(1, 0),
                    Tensor.concat(1, x, y), Tensor.stack(1, x, y), x.meanSquaredError(y, LossReduction.NONE),
                    x.categoricalCrossEntropyWithLogits(y, 1, LossReduction.NONE));
            for (Tensor output : outputs) assertRow(output, facts, batch, 0);
            assertEquals(DataType.INT64, outputs.get(14).descriptor().dataType());
            assertEquals(DataType.BOOL, condition.descriptor().dataType());
            assertEquals(List.of(0, 1, 2), fact(outputs.get(5), facts, batch).directInputs());
            assertEquals(List.of(0, 1), fact(x.add(x), facts, batch).directInputs());
            assertThrows(IllegalArgumentException.class, () -> fact(x.softmax(0), facts, batch));
            assertThrows(IllegalArgumentException.class, () -> fact(x.cumSum(0), facts, batch));
            assertThrows(IllegalArgumentException.class, () -> fact(x.mean(0), facts, batch));
        }
    }

    @Test void layoutAndIndexContributorMapsPreserveNonleadingRowsAndRejectArbitraryBatchAccess() {
        for (Dimension batch : batches()) {
            Tensor x = tensor(DataType.FLOAT64, fixed(2), batch, fixed(4));
            Tensor update = tensor(DataType.FLOAT64, fixed(2), batch, fixed(4));
            Tensor aligned = tensor(DataType.INT64, fixed(2), batch, fixed(2));
            Tensor table = tensor(DataType.FLOAT64, fixed(7), fixed(4));
            Tensor lookup = tensor(DataType.INT64, batch, fixed(4));
            Tensor batchedTable = tensor(DataType.FLOAT64, batch, fixed(7), fixed(4));
            Tensor tuples = tensor(DataType.INT64, batch, fixed(2), fixed(1));
            Tensor base = tensor(DataType.FLOAT64, batch, fixed(7));
            Tensor destinations = tensor(DataType.INT64, batch, fixed(2));
            Tensor values = tensor(DataType.FLOAT64, batch, fixed(2));
            var facts = declarations(batch, x, 1, update, 1, aligned, 1, table, -1, lookup, 0,
                    batchedTable, 0, tuples, 0, base, 0, destinations, 0, values, 0);
            for (Tensor result : List.of(x.reshape(Shape.ofDimensions(fixed(2), batch, fixed(2), fixed(2))),
                    x.sliceAxis(0, 0, 1), x.sliceUpdate(update, new long[] {0}, new int[] {0}, new long[] {1}),
                    x.cropToShape(Shape.ofDimensions(fixed(1), batch, fixed(4)), Shape.of(0, 0, 0)),
                    x.pad(new long[] {1, 0, 0}, new long[] {0, 0, 1}, ScalarValue.float64(0)),
                    x.tile(2, 1, 1), x.unfold(0, 1, 1), x.gatherElements(aligned, 2)))
                assertRow(result, facts, batch, 1);
            assertRow(x.select(0, 0), facts, batch, 0);
            assertRow(x.sum(0), facts, batch, 0);
            assertRow(table.gatherElements(lookup, 0), facts, batch, 0);
            assertRow(batchedTable.gatherNd(tuples, 1), facts, batch, 0);
            assertRow(base.scatterElements(destinations, values, 1), facts, batch, 0);
            assertThrows(IllegalArgumentException.class, () -> fact(batchedTable.gather(destinations, 0), facts, batch));
            if (batch.isStatic()) assertThrows(IllegalArgumentException.class, () -> fact(x.flip(1), facts, batch));
            assertThrows(IllegalArgumentException.class,
                    () -> fact(x.reshape(Shape.ofDimensions(batch, fixed(8))), facts, batch));
        }
    }

    @Test void explicitRowContributorsPermitOnlyOrdinarySharedSingletonAlignment() {
        Dimension batch = fixed(1);
        Tensor shared = tensor(DataType.FLOAT64, batch, fixed(4)), row = tensor(DataType.FLOAT64, batch, fixed(4));
        Tensor mask = tensor(DataType.BOOL, batch, fixed(4));
        Tensor data = tensor(DataType.FLOAT64, fixed(4), fixed(1));
        Tensor scale = tensor(DataType.FLOAT64, batch), bias = tensor(DataType.FLOAT64, fixed(1));
        Tensor table = tensor(DataType.FLOAT64, fixed(7), fixed(4)), indices = tensor(DataType.INT64, fixed(1));
        Tensor base = tensor(DataType.FLOAT64, batch, fixed(7));
        Tensor destinations = tensor(DataType.INT64, fixed(1), fixed(2)), updates = tensor(DataType.FLOAT64, fixed(1), fixed(2));
        var facts = declarations(batch, shared, -1, row, 0, mask, 0, data, -1, scale, 0, bias, -1,
                table, -1, indices, -1, base, 0, destinations, -1, updates, -1);
        assertRow(shared.sum(1, mask), facts, batch, 0);
        assertRow(shared.meanSquaredError(row, LossReduction.NONE), facts, batch, 0);
        assertRow(Tensor.where(mask, shared, shared), facts, batch, 0);
        assertRow(shared.sliceUpdate(row, new long[] {0}, new int[] {1}, new long[] {1}), facts, batch, 0);
        assertRow(Tensor.concat(1, shared, row), facts, batch, 0);
        assertRow(data.layerNorm(Shape.of(1), scale, bias, EPS), facts, batch, 1);
        assertRow(data.rmsNorm(Shape.of(1), scale, EPS), facts, batch, 1);
        assertRow(base.scatterElements(destinations, updates, 1), facts, batch, 0);
        assertTerminal(table.scatterAdd(indices, row, 0), facts, batch);
        assertTrue(Scan.declaredSemantics(shared.descriptor(), OptionalInt.empty(), batch).rowAxis().isEmpty());
    }

    @Test void nonleadingGroupedConvolutionClosureUsesBatchedMatmulAndExplicitAxisOneSum() {
        for (Dimension batch : batches()) {
            Tensor x = tensor(DataType.FLOAT64, batch, fixed(2), fixed(3), fixed(4));
            Tensor weight = tensor(DataType.FLOAT64, fixed(2), fixed(4), fixed(5));
            var facts = declarations(batch, x, 0, weight, -1);
            Tensor transposed = x.permute(1, 0, 2, 3);
            Tensor product = transposed.matmul(weight.expandDims(1));
            assertRow(transposed, facts, batch, 1);
            assertRow(product, facts, batch, 1);
            Scan.ValueSemantics terminal = fact(product.sum(1), facts, batch);
            assertTrue(terminal.rowAxis().isEmpty());
            assertTrue(terminal.terminal());
            assertThrows(IllegalArgumentException.class, () -> fact(product.mean(1), facts, batch));
            Tensor roundtrip = transposed.expandDims(0).squeeze(0).permute(1, 0, 2, 3);
            assertRow(roundtrip, facts, batch, 0);
        }
    }

    @Test void embeddingAndAllAddScatterFormsProduceTerminalSharedTargetsWithoutInventingRows() {
        for (Dimension batch : batches()) {
            Tensor table = tensor(DataType.FLOAT64, fixed(7), fixed(4));
            Tensor indices = tensor(DataType.INT64, batch);
            Tensor updates = tensor(DataType.FLOAT64, batch, fixed(4));
            Tensor nd = tensor(DataType.INT64, batch, fixed(1));
            Tensor aligned = tensor(DataType.INT64, batch, fixed(4));
            var facts = declarations(batch, table, -1, indices, 0, updates, 0, nd, 0, aligned, 0);
            assertRow(table.embedding(indices), facts, batch, 0);
            assertRow(table.gatherNd(nd), facts, batch, 0);
            assertRow(indices.oneHot(7), facts, batch, 0);
            Tensor scatter = table.scatterAdd(indices, updates, 0);
            assertTerminal(scatter, facts, batch);
            assertTerminal(scatter.scatterAdd(indices, updates, 0), facts, batch);
            assertTerminal(table.scatterElements(aligned, updates, 0, ScatterReduction.ADD), facts, batch);
            assertTerminal(table.scatterNd(nd, updates, ScatterReduction.ADD), facts, batch);
            assertThrows(IllegalArgumentException.class,
                    () -> fact(table.scatterElements(aligned, updates, 0, ScatterReduction.MUL), facts, batch));
            assertThrows(IllegalArgumentException.class,
                    () -> fact(updates.add(scatter.sum(0)), facts, batch));
        }
    }

    @Test void sharedScatterIndicesMayOmitUntouchedContributorAxes() {
        Dimension batch = fixed(3);
        Tensor base = tensor(DataType.FLOAT64, fixed(3), fixed(7));
        Tensor indices = tensor(DataType.INT32, fixed(2));
        Tensor updates = tensor(DataType.FLOAT64, batch, fixed(2));
        var facts = declarations(batch, base, -1, indices, -1, updates, 0);
        assertTerminal(base.scatterAdd(indices, updates, 1), facts, batch);

        Tensor suffixBase = tensor(DataType.FLOAT64, fixed(7), fixed(3));
        Tensor ndIndices = tensor(DataType.INT64, fixed(2), fixed(1));
        Tensor suffixUpdates = tensor(DataType.FLOAT64, fixed(2), batch);
        facts = declarations(batch, suffixBase, -1, ndIndices, -1, suffixUpdates, 1);
        assertTerminal(suffixBase.scatterNd(ndIndices, suffixUpdates, ScatterReduction.ADD), facts, batch);

        Tensor rowIndices = tensor(DataType.INT32, batch);
        Tensor independentUpdates = tensor(DataType.FLOAT64, batch, fixed(3));
        var axisFacts = declarations(batch, base, -1, rowIndices, 0, independentUpdates, 0);
        assertThrows(IllegalArgumentException.class,
                () -> fact(base.scatterAdd(rowIndices, independentUpdates, 1), axisFacts, batch));
        Tensor rowNdIndices = tensor(DataType.INT64, batch, fixed(1));
        Tensor independentSuffix = tensor(DataType.FLOAT64, fixed(3), batch);
        var ndFacts = declarations(batch, suffixBase, -1, rowNdIndices, 0, independentSuffix, 1);
        assertThrows(IllegalArgumentException.class,
                () -> fact(suffixBase.scatterNd(rowNdIndices, independentSuffix, ScatterReduction.ADD), ndFacts, batch));
    }

    @Test void batchNormHasFiveCanonicalRoutesAndSeparateStatisticCoordinates() {
        for (Dimension batch : batches()) {
            Tensor data = tensor(DataType.FLOAT64, fixed(3), batch);
            Tensor scale = tensor(DataType.FLOAT64, batch), bias = tensor(DataType.FLOAT64, batch);
            Tensor mean = tensor(DataType.FLOAT64, batch), variance = tensor(DataType.FLOAT64, batch);
            var facts = declarations(batch, data, 1, scale, 0, bias, 0, mean, 0, variance, 0);
            var result = data.batchNormTraining(1, scale, bias, mean, variance, ScalarValue.float64(0.1), EPS);
            TensorProducer p = result.output().provenance().orElseThrow().producer();
            List<List<Integer>> routes = List.of(List.of(0, 1, 2), List.of(0, 3), List.of(0, 4), List.of(0), List.of(0));
            for (int slot = 0; slot < 5; slot++) {
                assertRow(p.output(slot), facts, batch, slot == 0 ? 1 : 0);
                assertEquals(routes.get(slot), fact(p.output(slot), facts, batch).directInputs());
            }
        }
        Tensor x = tensor(DataType.FLOAT64, fixed(3), fixed(2));
        Tensor scale = tensor(DataType.FLOAT64, fixed(2)), bias = tensor(DataType.FLOAT64, fixed(2));
        Tensor mean = tensor(DataType.FLOAT64, fixed(2)), variance = tensor(DataType.FLOAT64, fixed(2));
        Dimension batch = fixed(3);
        var facts = declarations(batch, x, 0, scale, -1, bias, -1, mean, -1, variance, -1);
        var result = x.batchNormTraining(1, scale, bias, mean, variance, ScalarValue.float64(0.1), EPS);
        assertThrows(IllegalArgumentException.class, () -> fact(result.output(), facts, batch));
    }

    @Test void attentionWeightsExcludeValueRoutesButRetainValueBroadcastDomain() {
        for (Dimension batch : batches()) {
            Tensor q = tensor(DataType.FLOAT64, fixed(2), fixed(4));
            Tensor k = tensor(DataType.FLOAT64, fixed(3), fixed(4));
            Tensor v = tensor(DataType.FLOAT64, batch, fixed(3), fixed(5));
            var facts = declarations(batch, q, -1, k, -1, v, 0);
            var result = q.scaledDotProductAttentionWithWeights(k, v);
            assertRow(result.output(), facts, batch, 0);
            assertRow(result.weights(), facts, batch, 0);
            assertEquals(List.of(0, 1), fact(result.weights(), facts, batch).directInputs());
            Tensor features = tensor(DataType.FLOAT64, fixed(3), batch);
            facts.put(features, Scan.declaredSemantics(features.descriptor(), OptionalInt.of(1), batch));
            var featureResult = q.scaledDotProductAttentionWithWeights(k, features);
            assertRow(featureResult.output(), facts, batch, 1);
            assertTrue(fact(featureResult.weights(), facts, batch).rowAxis().isEmpty());
            Tensor mask = tensor(DataType.BOOL, fixed(2), fixed(3));
            facts.put(mask, Scan.declaredSemantics(mask.descriptor(), OptionalInt.empty(), batch));
            Tensor terminalValue = tensor(DataType.FLOAT64, fixed(3), fixed(5));
            facts.put(terminalValue, new Scan.ValueSemantics(OptionalInt.empty(), true, List.of()));
            var terminal = q.scaledDotProductAttentionWithWeights(k, terminalValue, mask);
            assertTerminal(terminal.output(), facts, batch);
            assertFalse(fact(terminal.weights(), facts, batch).terminal());
            assertEquals(List.of(0, 1, 3), fact(terminal.weights(), facts, batch).directInputs());
        }
    }

    @Test void convolutionPoolingAndWindowsPreserveTheirActualUnmixedCoordinates() {
        Dimension batch = fixed(3);
        Tensor image = tensor(DataType.FLOAT64, batch, fixed(2), fixed(4), fixed(4));
        Tensor weight = tensor(DataType.FLOAT64, fixed(4), fixed(2), fixed(1), fixed(1));
        Tensor volume = tensor(DataType.FLOAT64, batch, fixed(2), fixed(2), fixed(4), fixed(4));
        Tensor weight3 = tensor(DataType.FLOAT64, fixed(4), fixed(2), fixed(1), fixed(1), fixed(1));
        var facts = declarations(batch, image, 0, weight, -1, volume, 0, weight3, -1);
        var window = new Window2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false);
        var window3 = new Window3dAttrs(1, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
        Tensor columns = image.unfold2d(window), columns3 = volume.unfold3d(window3);
        for (Tensor result : List.of(image.conv2d(weight, Conv2dAttrs.defaults()),
                volume.conv3d(weight3, Conv3dAttrs.defaults()),
                image.maxPool2d(new MaxPool2dAttrs(2, 2, 2, 2, 0, 0, 1, 1, false)),
                image.averagePool2d(new AveragePool2dAttrs(2, 2, 2, 2, 0, 0, 1, 1, false)),
                volume.maxPool3d(new MaxPool3dAttrs(1, 2, 2, 1, 2, 2, 0, 0, 0, 1, 1, 1, false)),
                volume.averagePool3d(new AveragePool3dAttrs(1, 2, 2, 1, 2, 2, 0, 0, 0, 1, 1, 1, false)),
                columns, columns.fold2d(image.descriptor().shape(), window),
                columns3, columns3.fold3d(volume.descriptor().shape(), window3))) assertRow(result, facts, batch, 0);
        Tensor spatial = tensor(DataType.FLOAT64, fixed(1), fixed(1), batch, fixed(2));
        var spatialFacts = declarations(batch, spatial, 2);
        assertRow(spatial.maxPool2d(new MaxPool2dAttrs(1, 1, 1, 1, 0, 0, 1, 1, false)), spatialFacts, batch, 2);
        assertThrows(IllegalArgumentException.class, () -> fact(
                spatial.maxPool2d(new MaxPool2dAttrs(2, 1, 1, 1, 0, 0, 1, 1, false)), spatialFacts, batch));
    }

    @Test void foldChoosesLowestCompleteContributorWitnessAndRejectsMatchingExtentNeighborhoods() {
        var identity = new Window2dAttrs(1, 1, 1, 1, 0, 0, 1, 1, false);
        var featureWindow = new Window2dAttrs(1, 2, 1, 1, 0, 0, 1, 1, false);
        var identity3 = new Window3dAttrs(1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
        for (Dimension batch : List.of(fixed(1), fixed(3), new DynamicDimension("Rows"))) {
            Tensor columns = tensor(DataType.FLOAT64, fixed(1), fixed(1), batch);
            Tensor featureColumns = tensor(DataType.FLOAT64, fixed(1), fixed(2), batch);
            var facts = declarations(batch, columns, 2, featureColumns, 2);
            assertRow(columns.fold2d(Shape.ofDimensions(fixed(1), fixed(1), batch, fixed(1)), identity), facts, batch, 2);
            assertRow(featureColumns.fold2d(Shape.ofDimensions(fixed(1), fixed(1), batch, fixed(2)), featureWindow),
                    facts, batch, 2);
            assertRow(columns.fold3d(Shape.ofDimensions(fixed(1), fixed(1), fixed(1), batch, fixed(1)), identity3),
                    facts, batch, batch.equals(fixed(1)) ? 2 : 3);
        }
        var mixing = new Window2dAttrs(3, 1, 1, 1, 1, 0, 1, 1, false);
        for (long batch : new long[] {1, 3}) {
            Tensor columns = tensor(DataType.FLOAT64, fixed(1), fixed(3), fixed(batch));
            var facts = declarations(fixed(batch), columns, 2);
            Tensor mixedHeight = columns.fold2d(Shape.of(1, 1, batch, 1), mixing);
            if (batch == 1) assertRow(mixedHeight, facts, fixed(batch), 3);
            else assertThrows(IllegalArgumentException.class, () -> fact(mixedHeight, facts, fixed(batch)));
            Tensor allMixed = tensor(DataType.FLOAT64, fixed(1), fixed(9), fixed(batch));
            facts.put(allMixed, Scan.declaredSemantics(allMixed.descriptor(), OptionalInt.of(2), fixed(batch)));
            var mixingBoth = new Window2dAttrs(3, 3, 1, 1, 1, 1, 1, 1, false);
            assertThrows(IllegalArgumentException.class,
                    () -> fact(allMixed.fold2d(Shape.of(1, 1, batch, 1), mixingBoth), facts, fixed(batch)));
            Tensor vector = tensor(DataType.FLOAT64, fixed(batch));
            facts.put(vector, Scan.declaredSemantics(vector.descriptor(), OptionalInt.of(0), fixed(batch)));
            Tensor fullAxis = vector.unfold(0, batch, 2);
            assertRow(fullAxis, facts, fixed(batch), 1);
            assertRow(fullAxis.foldAxis(0, batch, 2), facts, fixed(batch), 0);
            Tensor image = tensor(DataType.FLOAT64, fixed(1), fixed(1), fixed(batch), fixed(1));
            facts.put(image, Scan.declaredSemantics(image.descriptor(), OptionalInt.of(2), fixed(batch)));
            var fullWindow = new Window2dAttrs(batch, 1, 2, 1, 0, 0, 1, 1, false);
            Tensor fullColumns = image.unfold2d(fullWindow);
            assertRow(fullColumns, facts, fixed(batch), 1);
            assertRow(fullColumns.fold2d(image.descriptor().shape(), fullWindow), facts, fixed(batch), batch == 1 ? 1 : 2);
        }
    }

    @Test void declarationsRoutesAndReshapesRequireCheckedBoundariesWithoutIdentities() {
        Dimension batch = fixed(1);
        var routes = new ArrayList<>(List.of(0, 1));
        var immutable = new Scan.ValueSemantics(OptionalInt.empty(), false, routes);
        routes.clear();
        assertEquals(List.of(0, 1), immutable.directInputs());
        assertThrows(UnsupportedOperationException.class, () -> immutable.directInputs().add(2));
        assertThrows(IllegalArgumentException.class, () -> new Scan.ValueSemantics(OptionalInt.of(0), true, List.of()));
        assertThrows(IllegalArgumentException.class, () -> new Scan.ValueSemantics(OptionalInt.empty(), false, List.of(1, 1)));
        Tensor x = tensor(DataType.FLOAT64, fixed(1), fixed(4));
        var facts = declarations(batch, x, 0);
        assertRow(x.reshape(Shape.of(1, 4)), facts, batch, 0);
        assertRow(x.reshape(Shape.of(1, 1, 4)), facts, batch, 0);
        assertThrows(IllegalArgumentException.class, () -> fact(x.reshape(Shape.of(4, 1)), facts, batch));
        var huge = new TensorDescriptor(DataType.FLOAT64,
                Shape.ofDimensions(fixed(0), fixed(Long.MAX_VALUE), fixed(2)), Optional.empty(), false);
        Tensor marker = tensor(DataType.BOOL);
        assertThrows(ArithmeticException.class, () -> Scan.declaredSemantics(huge, OptionalInt.of(0), fixed(0)));
        var named = new DynamicDimension("Rows");
        var repeated = new TensorDescriptor(DataType.FLOAT64, Shape.ofDimensions(named, named), Optional.empty(), false);
        assertThrows(IllegalArgumentException.class, () -> Scan.declaredSemantics(repeated, OptionalInt.of(0), named));
        assertEquals(marker.id().value() + 1, tensor(DataType.BOOL).id().value());
    }

    @Test void modelLengthSourceAdmitsTypedEmbeddingTerminalAndFeatureWorkBeforeAllocatingPorts() {
        for (Dimension batch : batches()) for (long time : new long[] {0, 2}) {
            Tensor h = tensor(DataType.FLOAT64, batch, fixed(4)), old = tensor(DataType.FLOAT64, fixed(7), fixed(4));
            Tensor x = tensor(DataType.FLOAT64, batch, fixed(4)), index = tensor(DataType.INT64, batch);
            Tensor base = tensor(DataType.FLOAT64, fixed(7), fixed(4));
            Tensor next = h.add(x).relu().softmax(1).cumSum(1).sliceAxis(1, 0, 4);
            Tensor terminal = base.scatterAdd(index, x, 0).scatterAdd(index, x, 0);
            var body = new ScanBody(List.of(h, old), List.of(x, index), List.of(base),
                    List.of(next, old.add(terminal)), List.of(next, index.oneHot(7)));
            var result = Scan.apply(List.of(tensor(DataType.FLOAT64, batch, fixed(4)), tensor(DataType.FLOAT64, fixed(7), fixed(4))),
                    List.of(tensor(DataType.FLOAT64, fixed(time), batch, fixed(4)), tensor(DataType.INT64, fixed(time), batch)),
                    List.of(tensor(DataType.FLOAT64, fixed(7), fixed(4))), tensor(DataType.INT64, batch), 1,
                    List.of(OptionalInt.of(0), OptionalInt.of(0)), body, false);
            assertEquals(DataType.BOOL, result.outputs().get(1).descriptor().dataType());
            assertEquals(Shape.of(7, 4), result.finalCarry().get(1).descriptor().shape());
        }
    }

    @Test void syntacticSharedCarryIsolationPrecedesBatchNormOutputSpecificRoutes() {
        Tensor h = tensor(DataType.FLOAT64, fixed(2), fixed(2)), x = tensor(DataType.FLOAT64, fixed(2), fixed(2));
        Tensor old = tensor(DataType.FLOAT64, fixed(2)), data = tensor(DataType.FLOAT64, fixed(3), fixed(2));
        Tensor scale = tensor(DataType.FLOAT64, fixed(2)), bias = tensor(DataType.FLOAT64, fixed(2));
        Tensor variance = tensor(DataType.FLOAT64, fixed(2));
        Tensor saved = data.batchNormTraining(1, scale, bias, old, variance, ScalarValue.float64(0.1), EPS)
                .output().provenance().orElseThrow().producer().output(3);
        var body = new ScanBody(List.of(h, old), List.of(x), List.of(data, scale, bias, variance),
                List.of(h, old.add(saved)), List.of());
        List<Tensor> actualCarry = List.of(tensor(DataType.FLOAT64, fixed(2), fixed(2)), tensor(DataType.FLOAT64, fixed(2)));
        List<Tensor> actualSequences = List.of(tensor(DataType.FLOAT64, fixed(0), fixed(2), fixed(2)));
        List<Tensor> actualInvariants = List.of(tensor(DataType.FLOAT64, fixed(3), fixed(2)), tensor(DataType.FLOAT64, fixed(2)),
                tensor(DataType.FLOAT64, fixed(2)), tensor(DataType.FLOAT64, fixed(2)));
        Tensor lengths = tensor(DataType.INT64, fixed(2)), marker = tensor(DataType.BOOL);
        assertThrows(IllegalArgumentException.class, () -> Scan.apply(actualCarry, actualSequences, actualInvariants,
                lengths, 1, List.of(OptionalInt.of(0)), body, false));
        assertEquals(marker.id().value() + 1, tensor(DataType.BOOL).id().value());
    }

    private static List<Dimension> batches() {
        return List.of(fixed(0), fixed(1), fixed(3), new DynamicDimension("Rows"));
    }
    private static StaticDimension fixed(long size) { return new StaticDimension(size); }
    private static Tensor tensor(DataType type, Dimension... dimensions) {
        Shape shape = Shape.ofDimensions(dimensions);
        return TensorFactory.create(new TensorDescriptor(type, shape,
                shape.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(), false));
    }
    private static IdentityHashMap<Tensor, Scan.ValueSemantics> declarations(Dimension batch, Object... pairs) {
        var facts = new IdentityHashMap<Tensor, Scan.ValueSemantics>();
        for (int i = 0; i < pairs.length; i += 2) {
            Tensor tensor = (Tensor) pairs[i];
            int row = (Integer) pairs[i + 1];
            facts.put(tensor, Scan.declaredSemantics(tensor.descriptor(), row < 0 ? OptionalInt.empty() : OptionalInt.of(row), batch));
        }
        return facts;
    }
    private static Scan.ValueSemantics fact(Tensor tensor, IdentityHashMap<Tensor, Scan.ValueSemantics> facts, Dimension batch) {
        var known = facts.get(tensor);
        if (known != null) return known;
        TensorProducer producer = tensor.provenance().orElseThrow().producer();
        var results = Scan.ordinarySemantics(producer.operation(), producer.inputs().stream().map(Tensor::descriptor).toList(),
                producer.inputs().stream().map(input -> fact(input, facts, batch)).toList(), producer.outputDescriptors(), batch);
        for (int slot = 0; slot < producer.outputCount(); slot++) facts.put(producer.output(slot), results.get(slot));
        return facts.get(tensor);
    }
    private static void assertRow(Tensor tensor, IdentityHashMap<Tensor, Scan.ValueSemantics> facts, Dimension batch, int row) {
        assertEquals(OptionalInt.of(row), fact(tensor, facts, batch).rowAxis(), tensor.provenance().orElseThrow().producer().operation().toString());
        assertFalse(fact(tensor, facts, batch).terminal());
    }
    private static void assertTerminal(Tensor tensor, IdentityHashMap<Tensor, Scan.ValueSemantics> facts, Dimension batch) {
        assertTrue(fact(tensor, facts, batch).rowAxis().isEmpty());
        assertTrue(fact(tensor, facts, batch).terminal());
    }
}

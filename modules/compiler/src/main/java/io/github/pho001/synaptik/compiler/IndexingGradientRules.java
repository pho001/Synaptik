package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.GatherNdAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;

/**
 * Builds first-order Gather and functional-scatter formulas through public Tensor operations.
 *
 * <p>Indices and coordinate tuples are immutable routing metadata and never receive a
 * cotangent. Gather routes data cotangents through matching additive scatter. Scatter routes
 * base and update cotangents according to its exact reduction: replacement masks the base,
 * addition preserves it, multiplication uses zero-count and safe-product facts, and extrema
 * share the cotangent among exact numeric-equality winners of the canonical forward result.
 * Duplicate updates remain separate candidates. Multiplication gives only a sole zero update the
 * product of the base and all non-zero updates, gives every update positive zero when a group has
 * several zeros, and preserves the exact forward update product for the base. Extrema include the
 * base and every update in their tie count; opposite signed zeros tie, while a canonical NaN
 * result matches no candidate and routes positive zero.</p>
 *
 * <p>The formulas do not inspect index values, storage, execution order, or backend behavior.
 * Preflight has already proved exact signatures, descriptors, normalized geometry, reduction
 * policy, canonical output identity where needed, and selected floating roles before this owner
 * may allocate a Tensor expression. Index bounds and replacement-target uniqueness remain
 * value-dependent validity obligations rather than compiler Shape constraints.</p>
 */
final class IndexingGradientRules {
    private IndexingGradientRules() {}

    /**
     * Builds input-position-aligned cotangents for one preflight-approved indexing occurrence.
     *
     * @param producer exact original Gather or scatter producer occurrence
     * @param gradient non-null accumulated cotangent for canonical output slot zero
     * @param selectedInputs non-null selected-route flags in original input order
     * @param constants non-null request-local exact typed logical-splat owner
     * @return a new array aligned with producer inputs; non-differentiable and unselected roles
     *     contain {@code null}
     * @throws IllegalStateException if the occurrence is outside the approved indexing matrix
     */
    static Tensor[] apply(
            TensorProducer producer,
            Tensor gradient,
            boolean[] selectedInputs,
            FirstOrderAutograd.DerivativeConstants constants) {
        var kind = producer.operation().kind();
        Tensor data = producer.inputs().getFirst();
        Tensor indices = producer.inputs().get(1);
        if (kind instanceof AxisGatherKind gatherKind) {
            int axis = ((IndexAxisAttrs) producer.operation().attrs()).axis();
            Tensor dataGradient = gatherKind == AxisGatherKind.GATHER
                    ? constants.zeroLike(data).scatterAdd(indices, gradient, axis)
                    : constants.zeroLike(data)
                            .scatterElements(indices, gradient, axis, ScatterReduction.ADD);
            return new Tensor[] {dataGradient, null};
        }
        if (kind == GatherNdKind.GATHER_ND) {
            int batch = ((GatherNdAttrs) producer.operation().attrs()).batchDimensions();
            return new Tensor[] {
                constants.zeroLike(data)
                        .scatterNd(indices, gradient, ScatterReduction.ADD, batch),
                null
            };
        }

        Tensor updates = producer.inputs().get(2);
        if (kind == AxisScatterKind.SCATTER_ADD) {
            int axis = ((IndexAxisAttrs) producer.operation().attrs()).axis();
            return new Tensor[] {
                selectedInputs[0] ? gradient : null,
                null,
                selectedInputs[2] ? gradient.gather(indices, axis) : null
            };
        }
        Geometry geometry;
        ScatterReduction reduction;
        if (kind == AxisScatterKind.SCATTER_ELEMENTS) {
            ScatterElementsAttrs attrs =
                    (ScatterElementsAttrs) producer.operation().attrs();
            geometry = new ElementsGeometry(indices, attrs.axis());
            reduction = attrs.reduction();
        } else if (kind == ScatterNdKind.SCATTER_ND) {
            ScatterNdAttrs attrs = (ScatterNdAttrs) producer.operation().attrs();
            geometry = new NdGeometry(indices, attrs.batchDimensions());
            reduction = attrs.reduction();
        } else {
            throw new IllegalStateException(
                    "indexing operation was not preflight-approved: " + producer.operation());
        }

        Tensor dataGradient = null;
        Tensor updatesGradient = null;
        if (reduction == ScatterReduction.NONE) {
            dataGradient = selectedInputs[0]
                    ? geometry.scatter(gradient, constants.zeroLike(updates), reduction)
                    : null;
            updatesGradient = selectedInputs[2] ? geometry.gather(gradient) : null;
        } else if (reduction == ScatterReduction.ADD) {
            dataGradient = selectedInputs[0] ? gradient : null;
            updatesGradient = selectedInputs[2] ? geometry.gather(gradient) : null;
        } else if (reduction == ScatterReduction.MUL) {
            Tensor[] result = multiplication(
                    data, updates, gradient, geometry, constants, selectedInputs);
            dataGradient = result[0];
            updatesGradient = result[1];
        } else if (reduction == ScatterReduction.MIN
                || reduction == ScatterReduction.MAX) {
            Tensor[] result = extrema(
                    producer.output(0),
                    data,
                    updates,
                    gradient,
                    geometry,
                    constants,
                    selectedInputs);
            dataGradient = result[0];
            updatesGradient = result[1];
        }
        return new Tensor[] {dataGradient, null, updatesGradient};
    }

    private static Tensor[] multiplication(
            Tensor data,
            Tensor updates,
            Tensor gradient,
            Geometry geometry,
            FirstOrderAutograd.DerivativeConstants constants,
            boolean[] selectedInputs) {
        Tensor zeroUpdates = constants.zeroLike(updates);
        Tensor oneUpdates = constants.oneLike(updates);
        Tensor zeroData = constants.zeroLike(data);
        Tensor oneData = constants.oneLike(data);
        Tensor isZero = updates.equalTo(zeroUpdates);
        Tensor zeroIndicator = Tensor.where(isZero, oneUpdates, zeroUpdates);
        Tensor safeUpdates = Tensor.where(isZero, oneUpdates, updates);
        Tensor updateZeroCount =
                geometry.scatter(zeroData, zeroIndicator, ScatterReduction.ADD);
        Tensor updateSafeProduct =
                geometry.scatter(oneData, safeUpdates, ScatterReduction.MUL);
        Tensor allUpdateProduct =
                geometry.scatter(oneData, updates, ScatterReduction.MUL);

        Tensor countAtUpdate = geometry.gather(updateZeroCount);
        Tensor safeProductAtUpdate = geometry.gather(updateSafeProduct);
        Tensor gradientAtUpdate = geometry.gather(gradient);
        Tensor dataAtUpdate = geometry.gather(data);
        Tensor safeDenominator = Tensor.where(isZero, oneUpdates, updates);
        Tensor regular = gradientAtUpdate
                .mul(dataAtUpdate)
                .mul(safeProductAtUpdate.div(safeDenominator));
        Tensor soleZero =
                gradientAtUpdate.mul(dataAtUpdate).mul(safeProductAtUpdate);
        Tensor updatesGradient = Tensor.where(
                countAtUpdate.equalTo(zeroUpdates),
                regular,
                Tensor.where(
                        countAtUpdate.equalTo(oneUpdates).logicalAnd(isZero),
                        soleZero,
                        zeroUpdates));
        return new Tensor[] {
            selectedInputs[0] ? gradient.mul(allUpdateProduct) : null,
            selectedInputs[2] ? updatesGradient : null
        };
    }

    private static Tensor[] extrema(
            Tensor output,
            Tensor data,
            Tensor updates,
            Tensor gradient,
            Geometry geometry,
            FirstOrderAutograd.DerivativeConstants constants,
            boolean[] selectedInputs) {
        Tensor zeroData = constants.zeroLike(data);
        Tensor oneData = constants.oneLike(data);
        Tensor zeroUpdates = constants.zeroLike(updates);
        Tensor oneUpdates = constants.oneLike(updates);
        Tensor baseMatches = data.equalTo(output);
        Tensor outputAtUpdate = geometry.gather(output);
        Tensor updateMatches = updates.equalTo(outputAtUpdate);
        Tensor baseIndicator = Tensor.where(baseMatches, oneData, zeroData);
        Tensor updateIndicator = Tensor.where(updateMatches, oneUpdates, zeroUpdates);
        Tensor tieCount =
                geometry.scatter(baseIndicator, updateIndicator, ScatterReduction.ADD);
        Tensor hasWinner = tieCount.greaterThan(zeroData);
        Tensor safeTieCount = Tensor.where(hasWinner, tieCount, oneData);
        Tensor shared = Tensor.where(
                hasWinner, gradient.div(safeTieCount), zeroData);
        return new Tensor[] {
            selectedInputs[0] ? Tensor.where(baseMatches, shared, zeroData) : null,
            selectedInputs[2]
                    ? Tensor.where(updateMatches, geometry.gather(shared), zeroUpdates)
                    : null
        };
    }

    private sealed interface Geometry permits ElementsGeometry, NdGeometry {
        Tensor gather(Tensor value);

        Tensor scatter(Tensor base, Tensor updates, ScatterReduction reduction);
    }

    private record ElementsGeometry(Tensor indices, int axis) implements Geometry {
        @Override
        public Tensor gather(Tensor value) {
            return value.gatherElements(indices, axis);
        }

        @Override
        public Tensor scatter(Tensor base, Tensor updates, ScatterReduction reduction) {
            return base.scatterElements(indices, updates, axis, reduction);
        }
    }

    private record NdGeometry(Tensor indices, int batchDimensions) implements Geometry {
        @Override
        public Tensor gather(Tensor value) {
            return value.gatherNd(indices, batchDimensions);
        }

        @Override
        public Tensor scatter(Tensor base, Tensor updates, ScatterReduction reduction) {
            return base.scatterNd(indices, updates, reduction, batchDimensions);
        }
    }
}

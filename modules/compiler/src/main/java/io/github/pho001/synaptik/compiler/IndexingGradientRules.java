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

import java.util.function.Function;

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
 *
 * <p>Descriptor access preserves original source roles without resolving primal values.
 * Real construction uses Tensor references directly; checked metadata construction retains
 * the same eager operation sites, including temporaries not used by a returned cotangent.</p>
 *
 * @param <V> direct Tensor or checked metadata operand representation
 */
final class IndexingGradientRules<V> {
    private final OrdinaryConstruction<V> ops;

    /**
     * Binds this formula owner to one compile-local construction route.
     * @param ops non-null direct Tensor or checked metadata construction
     */
    IndexingGradientRules(OrdinaryConstruction<V> ops) {
        this.ops = java.util.Objects.requireNonNull(ops);
    }

    /**
     * Builds input-position-aligned cotangents for one preflight-approved indexing occurrence.
     *
     * @param producer exact original Gather or scatter producer occurrence
     * @param gradient non-null accumulated cotangent for canonical output slot zero
     * @param selectedInputs non-null selected-route flags in original input order
     * @param primal non-null resolver from original primal wrappers to lexical values
     * @return a new array aligned with producer inputs; non-differentiable and unselected roles
     *     contain {@code null}
     * @throws IllegalStateException if the occurrence is outside the approved indexing matrix
     */
    V[] apply(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            boolean[] selectedInputs,
            Function<V, V> primal) {
        var kind = producer.operation().kind();
        V data = producer.inputs().getFirst();
        V indices = primal.apply(producer.inputs().get(1));
        if (kind instanceof AxisGatherKind gatherKind) {
            int axis = ((IndexAxisAttrs) producer.operation().attrs()).axis();
            V dataGradient = gatherKind == AxisGatherKind.GATHER
                    ? ops.scatterAdd(ops.contiguous(ops.zeroLike(data)), indices, gradient, axis)
                    : ops.scatterElements(ops.zeroLike(data), indices, gradient, axis, ScatterReduction.ADD);
            return ops.result(dataGradient, null);
        }
        if (kind == GatherNdKind.GATHER_ND) {
            int batch = ((GatherNdAttrs) producer.operation().attrs()).batchDimensions();
            return ops.result(ops.scatterNd(ops.zeroLike(data), indices, gradient, ScatterReduction.ADD, batch),
                null);
        }

        V updates = producer.inputs().get(2);
        if (kind == AxisScatterKind.SCATTER_ADD) {
            int axis = ((IndexAxisAttrs) producer.operation().attrs()).axis();
            return ops.result(selectedInputs[0] ? gradient : null,
                null,
                selectedInputs[2] ? ops.gather(gradient, indices, axis) : null);
        }
        Geometry<V> geometry;
        ScatterReduction reduction;
        if (kind == AxisScatterKind.SCATTER_ELEMENTS) {
            ScatterElementsAttrs attrs =
                    (ScatterElementsAttrs) producer.operation().attrs();
            geometry = new ElementsGeometry<>(indices, attrs.axis(), ops);
            reduction = attrs.reduction();
        } else if (kind == ScatterNdKind.SCATTER_ND) {
            ScatterNdAttrs attrs = (ScatterNdAttrs) producer.operation().attrs();
            geometry = new NdGeometry<>(indices, attrs.batchDimensions(), ops);
            reduction = attrs.reduction();
        } else {
            throw new IllegalStateException(
                    "indexing operation was not preflight-approved: " + producer.operation());
        }

        V dataGradient = null;
        V updatesGradient = null;
        if (reduction == ScatterReduction.NONE) {
            dataGradient = selectedInputs[0]
                    ? geometry.scatter(gradient, ops.zeroLike(updates), reduction)
                    : null;
            updatesGradient = selectedInputs[2] ? geometry.gather(gradient) : null;
        } else if (reduction == ScatterReduction.ADD) {
            dataGradient = selectedInputs[0] ? gradient : null;
            updatesGradient = selectedInputs[2] ? geometry.gather(gradient) : null;
        } else if (reduction == ScatterReduction.MUL) {
            V[] result = multiplication(
                    data, updates, gradient, geometry, selectedInputs, primal);
            dataGradient = result[0];
            updatesGradient = result[1];
        } else if (reduction == ScatterReduction.MIN
                || reduction == ScatterReduction.MAX) {
            V[] result = extrema(
                    producer.output(0),
                    data,
                    updates,
                    gradient,
                    geometry,
                    selectedInputs,
                    primal);
            dataGradient = result[0];
            updatesGradient = result[1];
        }
        return ops.result(dataGradient, null, updatesGradient);
    }

    private V[] multiplication(
            V data,
            V updates,
            V gradient,
            Geometry<V> geometry,
            boolean[] selectedInputs,
            Function<V, V> primal) {
        V zeroUpdates = ops.zeroLike(updates);
        V oneUpdates = ops.oneLike(updates);
        V zeroData = ops.zeroLike(data);
        V oneData = ops.oneLike(data);
        V dataValue = primal.apply(data);
        V updatesValue = primal.apply(updates);
        V isZero = ops.equalTo(updatesValue, zeroUpdates);
        V zeroIndicator = ops.where(isZero, oneUpdates, zeroUpdates);
        V safeUpdates = ops.where(isZero, oneUpdates, updatesValue);
        V updateZeroCount =
                geometry.scatter(zeroData, zeroIndicator, ScatterReduction.ADD);
        V updateSafeProduct =
                geometry.scatter(oneData, safeUpdates, ScatterReduction.MUL);
        V allUpdateProduct =
                geometry.scatter(oneData, updatesValue, ScatterReduction.MUL);
        
        V countAtUpdate = geometry.gather(updateZeroCount);
        V safeProductAtUpdate = geometry.gather(updateSafeProduct);
        V gradientAtUpdate = geometry.gather(gradient);
        V dataAtUpdate = geometry.gather(dataValue);
        V safeDenominator = ops.where(isZero, oneUpdates, updatesValue);
        V regular = ops.mul(ops.mul(gradientAtUpdate, dataAtUpdate), ops.div(safeProductAtUpdate, safeDenominator));
        V soleZero =
                ops.mul(ops.mul(gradientAtUpdate, dataAtUpdate), safeProductAtUpdate);
        V updatesGradient = ops.where(ops.equalTo(countAtUpdate, zeroUpdates), regular, ops.where(ops.logicalAnd(ops.equalTo(countAtUpdate, oneUpdates), isZero), soleZero, zeroUpdates));
        return ops.result(selectedInputs[0] ? ops.mul(gradient, allUpdateProduct) : null,
            selectedInputs[2] ? updatesGradient : null);
    }

    private V[] extrema(
            V output,
            V data,
            V updates,
            V gradient,
            Geometry<V> geometry,
            boolean[] selectedInputs,
            Function<V, V> primal) {
        V zeroData = ops.zeroLike(data);
        V oneData = ops.oneLike(data);
        V zeroUpdates = ops.zeroLike(updates);
        V oneUpdates = ops.oneLike(updates);
        V outputValue = primal.apply(output);
        V baseMatches = ops.equalTo(primal.apply(data), outputValue);
        V outputAtUpdate = geometry.gather(outputValue);
        V updateMatches = ops.equalTo(primal.apply(updates), outputAtUpdate);
        V baseIndicator = ops.where(baseMatches, oneData, zeroData);
        V updateIndicator = ops.where(updateMatches, oneUpdates, zeroUpdates);
        V tieCount =
                geometry.scatter(baseIndicator, updateIndicator, ScatterReduction.ADD);
        V hasWinner = ops.greaterThan(tieCount, zeroData);
        V safeTieCount = ops.where(hasWinner, tieCount, oneData);
        V shared = ops.where(hasWinner, ops.div(gradient, safeTieCount), zeroData);
        return ops.result(selectedInputs[0] ? ops.where(baseMatches, shared, zeroData) : null,
            selectedInputs[2]
                    ? ops.where(updateMatches, geometry.gather(shared), zeroUpdates)
                    : null);
    }

    /**
     * Retains the original scatter routing choice for both construction representations.
     * Indices are resolved once by the caller; geometry never resolves descriptor-only roles.
     * @param <V> direct Tensor or metadata operand type
     */
    private sealed interface Geometry<V> permits ElementsGeometry, NdGeometry {
        /**
         * Reads the same selected positions used by the source scatter.
         * @param value data operand to route
         * @return selected values in update-position order
         */
        V gather(V value);

        /**
         * Constructs the source scatter geometry with its existing reduction policy.
         * @param base exact base operand
         * @param updates update-position-aligned contributions
         * @param reduction existing source or derivative routing policy
         * @return ordinary functional scatter result
         */
        V scatter(V base, V updates, ScatterReduction reduction);
    }

    /**
     * Shares the existing axis-element routing without copying shape or index algorithms.
     * @param <V> direct operand type
     * @param indices already resolved exact index operand
     * @param axis canonical source axis
     * @param construction compile-local operation construction
     */
    private record ElementsGeometry<V>(
            V indices, int axis, OrdinaryConstruction<V> construction) implements Geometry<V> {
        @Override
        public V gather(V value) {
            return construction.gatherElements(value, indices, axis);
        }

        @Override
        public V scatter(V base, V updates, ScatterReduction reduction) {
            return construction.scatterElements(base, indices, updates, axis, reduction);
        }
    }

    /**
     * Shares the existing tuple routing without copying coordinate or bounds algorithms.
     * @param <V> direct operand type
     * @param indices already resolved exact coordinate tuples
     * @param batchDimensions canonical source batch prefix
     * @param construction compile-local operation construction
     */
    private record NdGeometry<V>(
            V indices, int batchDimensions,
            OrdinaryConstruction<V> construction) implements Geometry<V> {
        @Override
        public V gather(V value) {
            return construction.gatherNd(value, indices, batchDimensions);
        }

        @Override
        public V scatter(V base, V updates, ScatterReduction reduction) {
            return construction.scatterNd(base, indices, updates, reduction, batchDimensions);
        }
    }
}

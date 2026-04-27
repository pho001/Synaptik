package backend.cpu.kernels.linalg;

import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;

final class ScaledDotProductAttentionRuntimeCache {
    private final Tensor weights;
    private Tensor queryGrad;
    private Tensor keyGrad;
    private Tensor valueGrad;
    private Tensor dWeights;
    private Tensor dScores;
    private boolean backwardGradsValid;

    ScaledDotProductAttentionRuntimeCache(Tensor weights) {
        this.weights = weights;
        this.backwardGradsValid = false;
    }

    Tensor weights() {
        return weights;
    }

    Tensor queryGrad() {
        return queryGrad;
    }

    boolean hasBackwardGrads() {
        return backwardGradsValid && queryGrad != null && keyGrad != null && valueGrad != null;
    }

    void setQueryGrad(Tensor queryGrad) {
        this.queryGrad = queryGrad;
    }

    Tensor keyGrad() {
        return keyGrad;
    }

    void setKeyGrad(Tensor keyGrad) {
        this.keyGrad = keyGrad;
    }

    Tensor valueGrad() {
        return valueGrad;
    }

    void setValueGrad(Tensor valueGrad) {
        this.valueGrad = valueGrad;
    }

    Tensor requireQueryGrad(int[] shape, DataType dataType) {
        queryGrad = ensureTensor(queryGrad, shape, dataType, "attention_query_grad_cache");
        return queryGrad;
    }

    Tensor requireKeyGrad(int[] shape, DataType dataType) {
        keyGrad = ensureTensor(keyGrad, shape, dataType, "attention_key_grad_cache");
        return keyGrad;
    }

    Tensor requireValueGrad(int[] shape, DataType dataType) {
        valueGrad = ensureTensor(valueGrad, shape, dataType, "attention_value_grad_cache");
        return valueGrad;
    }

    Tensor requireDWeights(int[] shape, DataType dataType) {
        dWeights = ensureTensor(dWeights, shape, dataType, "attention_dweights");
        return dWeights;
    }

    Tensor requireDScores(int[] shape, DataType dataType) {
        dScores = ensureTensor(dScores, shape, dataType, "attention_dscores");
        return dScores;
    }

    void markBackwardGradsReady() {
        backwardGradsValid = true;
    }

    void resetForNextExecution() {
        backwardGradsValid = false;
    }

    private static Tensor ensureTensor(Tensor current, int[] shape, DataType dataType, String label) {
        if (current != null && current.getDataType() == dataType && Arrays.equals(current.getShapeUnsafe(), shape)) {
            return current;
        }
        return new Tensor(shape.clone(), List.of(), label, dataType);
    }
}

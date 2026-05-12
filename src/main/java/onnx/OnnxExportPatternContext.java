package onnx;

import tensor.Tensor;

import java.util.IdentityHashMap;

record OnnxExportPatternContext(
        Tensor graphOutput,
        IdentityHashMap<Tensor, Integer> consumerCounts,
        IdentityHashMap<Tensor, Integer> ids,
        OnnxNameRegistry names
) {
    boolean canConsume(Tensor tensor) {
        return tensor != graphOutput && consumerCounts.getOrDefault(tensor, 0) == 1;
    }

    int id(Tensor tensor) {
        return ids.get(tensor);
    }

    String name(Tensor tensor) {
        return names.nameFor(tensor, id(tensor));
    }
}

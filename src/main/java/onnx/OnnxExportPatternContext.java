package onnx;

import tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.Set;

record OnnxExportPatternContext(
        Set<Tensor> graphOutputs,
        IdentityHashMap<Tensor, Integer> consumerCounts,
        IdentityHashMap<Tensor, Integer> ids,
        OnnxNameRegistry names
) {
    boolean canConsume(Tensor tensor) {
        return !graphOutputs.contains(tensor) && consumerCounts.getOrDefault(tensor, 0) == 1;
    }

    int id(Tensor tensor) {
        return ids.get(tensor);
    }

    String name(Tensor tensor) {
        return names.nameFor(tensor, id(tensor));
    }

    String auxiliary(String baseName) {
        return names.auxiliary(baseName);
    }
}

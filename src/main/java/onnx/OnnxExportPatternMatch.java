package onnx;

import tensor.Tensor;

import java.util.List;
import java.util.Set;

record OnnxExportPatternMatch(
        OnnxProto.NodeProto node,
        Set<Tensor> consumedTensors,
        List<OnnxProto.TensorProto> initializers
) {
    OnnxExportPatternMatch(OnnxProto.NodeProto node, Set<Tensor> consumedTensors) {
        this(node, consumedTensors, List.of());
    }

    OnnxExportPatternMatch {
        initializers = List.copyOf(initializers);
    }
}

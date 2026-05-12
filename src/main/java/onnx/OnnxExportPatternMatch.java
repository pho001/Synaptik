package onnx;

import tensor.Tensor;

import java.util.Set;

record OnnxExportPatternMatch(
        OnnxProto.NodeProto node,
        Set<Tensor> consumedTensors
) {
}

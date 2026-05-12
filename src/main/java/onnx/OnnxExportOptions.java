package onnx;

/**
 * Export policy for the first ONNX interchange subset.
 */
public record OnnxExportOptions(
        int opsetVersion,
        String producerName,
        String graphName,
        OnnxLeafTensorPolicy leafTensorPolicy
) {
    public static final int DEFAULT_OPSET = 18;

    public OnnxExportOptions {
        if (opsetVersion <= 0) {
            throw new IllegalArgumentException("opsetVersion must be positive.");
        }
        producerName = producerName == null || producerName.isBlank() ? "synaptik" : producerName;
        graphName = graphName == null || graphName.isBlank() ? "synaptik_graph" : graphName;
        leafTensorPolicy = leafTensorPolicy == null ? OnnxLeafTensorPolicy.INPUTS : leafTensorPolicy;
    }

    public static OnnxExportOptions defaults() {
        return new OnnxExportOptions(DEFAULT_OPSET, "synaptik", "synaptik_graph", OnnxLeafTensorPolicy.INPUTS);
    }

    public OnnxExportOptions withLeafTensorPolicy(OnnxLeafTensorPolicy policy) {
        return new OnnxExportOptions(opsetVersion, producerName, graphName, policy);
    }
}

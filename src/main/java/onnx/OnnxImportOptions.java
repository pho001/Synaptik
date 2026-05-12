package onnx;

/**
 * Import policy for ONNX models.
 */
public record OnnxImportOptions(
        int minimumOpsetVersion,
        int maximumOpsetVersion
) {
    public OnnxImportOptions {
        if (minimumOpsetVersion <= 0 || maximumOpsetVersion < minimumOpsetVersion) {
            throw new IllegalArgumentException("Invalid ONNX opset version range.");
        }
    }

    public static OnnxImportOptions defaults() {
        return new OnnxImportOptions(7, OnnxExportOptions.DEFAULT_OPSET);
    }
}

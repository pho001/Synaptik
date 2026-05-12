package onnx;

/**
 * Raised when an ONNX model uses syntax outside the supported interchange subset.
 */
public final class OnnxUnsupportedException extends OnnxException {
    public OnnxUnsupportedException(String message) {
        super(message);
    }
}

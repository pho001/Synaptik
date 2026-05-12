package onnx;

/**
 * Base exception for ONNX import/export failures.
 */
public class OnnxException extends RuntimeException {
    public OnnxException(String message) {
        super(message);
    }

    public OnnxException(String message, Throwable cause) {
        super(message, cause);
    }
}

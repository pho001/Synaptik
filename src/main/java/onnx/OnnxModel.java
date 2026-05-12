package onnx;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * In-memory ONNX model wrapper.
 */
public record OnnxModel(OnnxProto.ModelProto proto) {
    public OnnxModel {
        proto = Objects.requireNonNull(proto, "proto cannot be null");
    }

    public void write(Path path) {
        Objects.requireNonNull(path, "path cannot be null");
        try {
            Path parent = path.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            try (OutputStream out = Files.newOutputStream(path)) {
                proto.writeTo(out);
            }
        } catch (IOException e) {
            throw new OnnxException("Failed to write ONNX model to " + path + ".", e);
        }
    }

    public byte[] toByteArray() {
        return proto.toByteArray();
    }
}

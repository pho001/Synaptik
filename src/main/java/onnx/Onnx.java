package onnx;

import tensor.Tensor;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Public ONNX interchange facade.
 */
public final class Onnx {
    private Onnx() {
    }

    public static OnnxModel exportModel(Tensor output) {
        return exportModel(output, OnnxExportOptions.defaults());
    }

    public static OnnxModel exportModel(Tensor output, OnnxExportOptions options) {
        return new OnnxModel(OnnxGraphExporter.export(output, options));
    }

    public static void write(Tensor output, Path path) {
        write(output, path, OnnxExportOptions.defaults());
    }

    public static void write(Tensor output, Path path, OnnxExportOptions options) {
        exportModel(output, options).write(path);
    }

    public static ImportedOnnxModel read(Path path) {
        return read(path, OnnxImportOptions.defaults());
    }

    public static ImportedOnnxModel read(Path path, OnnxImportOptions options) {
        Objects.requireNonNull(path, "path cannot be null");
        try (InputStream in = Files.newInputStream(path)) {
            return importModel(OnnxProto.ModelProto.parseFrom(in), options);
        } catch (IOException e) {
            throw new OnnxException("Failed to read ONNX model from " + path + ".", e);
        }
    }

    public static ImportedOnnxModel importModel(OnnxProto.ModelProto model) {
        return importModel(model, OnnxImportOptions.defaults());
    }

    public static ImportedOnnxModel importModel(OnnxProto.ModelProto model, OnnxImportOptions options) {
        return OnnxGraphImporter.importModel(model, options);
    }
}

package onnx;

import config.compile.CompileConfig;
import graph.CompiledGraph;
import tensor.CompileMode;
import tensor.Tensor;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Imported ONNX graph expressed as regular Synaptik tensors.
 */
public final class ImportedOnnxModel {
    private final OnnxProto.ModelProto source;
    private final Map<String, Tensor> inputs;
    private final Map<String, Tensor> outputs;

    ImportedOnnxModel(OnnxProto.ModelProto source, Map<String, Tensor> inputs, Map<String, Tensor> outputs) {
        this.source = Objects.requireNonNull(source, "source cannot be null");
        this.inputs = Collections.unmodifiableMap(new LinkedHashMap<>(inputs == null ? Map.of() : inputs));
        this.outputs = Collections.unmodifiableMap(new LinkedHashMap<>(outputs == null ? Map.of() : outputs));
    }

    public OnnxProto.ModelProto source() {
        return source;
    }

    public Map<String, Tensor> inputs() {
        return inputs;
    }

    public Map<String, Tensor> outputs() {
        return outputs;
    }

    public Tensor input(String name) {
        Tensor tensor = inputs.get(name);
        if (tensor == null) {
            throw new IllegalArgumentException("Unknown ONNX input: " + name);
        }
        return tensor;
    }

    public Tensor output(String name) {
        Tensor tensor = outputs.get(name);
        if (tensor == null) {
            throw new IllegalArgumentException("Unknown ONNX output: " + name);
        }
        return tensor;
    }

    public CompiledGraph compile(CompileConfig config) {
        if (outputs.size() != 1) {
            throw new IllegalStateException("compile(config) requires exactly one ONNX output; got " + outputs.keySet());
        }
        return compile(outputs.keySet().iterator().next(), config);
    }

    public CompiledGraph compile(String outputName, CompileConfig config) {
        return CompiledGraph.compile(output(outputName), config, CompileMode.INFERENCE_ONLY);
    }
}

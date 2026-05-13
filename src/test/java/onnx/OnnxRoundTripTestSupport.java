package onnx;

import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import tensor.Tensor;

import java.util.Map;

final class OnnxRoundTripTestSupport {
    private OnnxRoundTripTestSupport() {
    }

    static ImportedOnnxModel exportImport(Tensor output) {
        return Onnx.importModel(Onnx.exportModel(
                output,
                OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS)
        ).proto());
    }

    static double[] executeRoundTrip(Tensor output, String outputName, Map<String, ?> inputs) {
        ImportedOnnxModel imported = exportImport(output);
        for (Map.Entry<String, ?> entry : inputs.entrySet()) {
            setInput(imported, entry.getKey(), entry.getValue());
        }
        imported.compile(outputName, CompileConfig.inference())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        return imported.output(outputName).toDoubleArrayCopy();
    }

    static boolean[] executeRoundTripBool(Tensor output, String outputName, Map<String, ?> inputs) {
        ImportedOnnxModel imported = exportImport(output);
        for (Map.Entry<String, ?> entry : inputs.entrySet()) {
            setInput(imported, entry.getKey(), entry.getValue());
        }
        imported.compile(outputName, CompileConfig.inference())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        return imported.output(outputName).toBooleanArrayCopy();
    }

    static double[] executeImported(ImportedOnnxModel imported, String outputName) {
        imported.compile(outputName, CompileConfig.inference())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        return imported.output(outputName).toDoubleArrayCopy();
    }

    private static void setInput(ImportedOnnxModel imported, String name, Object value) {
        if (value instanceof float[] floats) {
            imported.input(name).setData(floats);
        } else if (value instanceof double[] doubles) {
            imported.input(name).setData(doubles);
        } else if (value instanceof int[] ints) {
            imported.input(name).setData(ints);
        } else if (value instanceof byte[] bytes) {
            imported.input(name).setData(bytes);
        } else {
            throw new IllegalArgumentException("Unsupported ONNX round-trip input type: " + value.getClass());
        }
    }
}

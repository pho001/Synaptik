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

    static double[] executeRoundTrip(Tensor output, String outputName, Map<String, float[]> inputs) {
        ImportedOnnxModel imported = exportImport(output);
        for (Map.Entry<String, float[]> entry : inputs.entrySet()) {
            imported.input(entry.getKey()).setData(entry.getValue());
        }
        imported.compile(outputName, CompileConfig.inference())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        return imported.output(outputName).toDoubleArrayCopy();
    }

    static double[] executeImported(ImportedOnnxModel imported, String outputName) {
        imported.compile(outputName, CompileConfig.inference())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        return imported.output(outputName).toDoubleArrayCopy();
    }
}

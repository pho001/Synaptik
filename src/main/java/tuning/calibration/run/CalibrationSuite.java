package tuning.calibration.run;

import backend.ComputeBackend;
import backend.accelerator.select.AcceleratorRuntimeAvailability;
import tensor.DataType;
import tuning.calibration.PlatformCalibrationDefaults;
import tuning.calibration.PlatformCalibrationStep;
import tuning.calibration.family.CalibrationFamilyId;
import tuning.preset.TuningPreset;

import java.util.List;

public final class CalibrationSuite {
    private CalibrationSuite() {
    }

    public static List<PlatformCalibrationStep> stepsFor(
            CalibrationFamilyId family,
            String name,
            TuningPreset preset,
            DataType dataType
    ) {
        return switch (family) {
            case SCHEDULER -> List.of(PlatformCalibrationDefaults.schedulerStep(name, preset));
            case MATMUL -> List.of(
                    PlatformCalibrationDefaults.matmulJavaStep(name + "-java", preset),
                    PlatformCalibrationDefaults.matmulBlasDispatchStep(name + "-blas", preset),
                    PlatformCalibrationDefaults.matmulBlasWideDispatchStep(name + "-blas-wide", preset)
            );
            case ATTENTION_MATMUL -> List.of(PlatformCalibrationDefaults.attentionMatmulStep(name, preset));
            case ELEMENTWISE_DISPATCH -> List.of(PlatformCalibrationDefaults.elementwiseDispatchStep(name, preset));
            case FUSED_DISPATCH -> List.of(PlatformCalibrationDefaults.fusedDispatchStep(name, preset));
            case FUSED_ASM_WIDTH -> List.of(PlatformCalibrationDefaults.fusedAsmWidthStep(name, preset, dataType));
            case REDUCTION -> List.of(PlatformCalibrationDefaults.reductionStep(name, preset));
            case ATTENTION_THRESHOLDS -> List.of(PlatformCalibrationDefaults.attentionStep(name, preset));
            case MATERIALIZATION -> List.of(
                    PlatformCalibrationDefaults.materializationStep(name, preset),
                    PlatformCalibrationDefaults.whereMaterializationStep(name + "-where", preset)
            );
            case METAL_SELECTION -> {
                if (dataType != DataType.FLOAT32) {
                    throw new IllegalArgumentException("metal-selection calibration currently supports f32 only.");
                }
                if (!AcceleratorRuntimeAvailability.isAvailable(ComputeBackend.GPU_METAL)) {
                    throw new IllegalArgumentException("metal-selection calibration requires an available Metal runtime.");
                }
                yield List.of(PlatformCalibrationDefaults.acceleratorMetalSelectionStep(name, preset));
            }
        };
    }
}

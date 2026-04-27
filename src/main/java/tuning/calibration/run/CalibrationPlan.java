package tuning.calibration.run;

import tensor.DataType;
import tuning.calibration.PlatformCalibrationStep;
import tuning.calibration.family.CalibrationFamilyId;
import tuning.calibration.family.CalibrationFamilyRegistry;

import java.util.ArrayList;
import java.util.List;

public record CalibrationPlan(
        DataType dataType,
        int passIndex,
        int passCount,
        List<PlatformCalibrationStep> steps
) {
    public CalibrationPlan {
        steps = steps == null ? List.of() : List.copyOf(steps);
    }

    public static CalibrationPlan build(CalibrationCommand command, DataType dataType, int passIndex) {
        List<PlatformCalibrationStep> steps = new ArrayList<>();
        for (CalibrationFamilyId family : command.families()) {
            if (!CalibrationFamilyRegistry.supportsDType(family, dataType)) {
                throw new IllegalArgumentException("Family " + family + " does not support dtype " + dataType);
            }
            String name = "calib-" + CalibrationCommand.dtypeId(dataType) + "-"
                    + CalibrationFamilyRegistry.spec(family).cliName()
                    + "-p" + passIndex;
            steps.addAll(CalibrationSuite.stepsFor(family, name, command.preset(), dataType));
        }
        return new CalibrationPlan(dataType, passIndex, command.passCount(), steps);
    }
}

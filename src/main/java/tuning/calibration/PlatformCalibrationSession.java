package tuning.calibration;

import tuning.measure.DefaultMeasurementEngine;
import tuning.measure.MeasurementEngine;
import tuning.store.JsonFilePlatformRuntimeProfileStore;
import tuning.store.PlatformRuntimeProfileStore;
import tuning.validate.DefaultValidationEngine;
import tuning.validate.ValidationEngine;

public interface PlatformCalibrationSession {
    PlatformCalibrationResult run();

    static PlatformCalibrationSession create(PlatformCalibrationRequest request) {
        return create(
                request,
                new DefaultMeasurementEngine(),
                new DefaultValidationEngine(),
                new JsonFilePlatformRuntimeProfileStore(request.seedRuntimeProfile())
        );
    }

    static PlatformCalibrationSession create(
            PlatformCalibrationRequest request,
            MeasurementEngine measurementEngine,
            ValidationEngine validationEngine,
            PlatformRuntimeProfileStore profileStore
    ) {
        return new DefaultPlatformCalibrationSession(request, measurementEngine, validationEngine, profileStore);
    }
}

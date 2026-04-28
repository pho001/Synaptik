package tuning.calibration;

import tuning.measure.DefaultMeasurementEngine;
import tuning.measure.MeasurementEngine;
import tuning.store.JsonFilePlatformRuntimeProfileStore;
import tuning.store.PlatformRuntimeProfileStore;
import tuning.validate.DefaultValidationEngine;
import tuning.validate.ValidationEngine;

/**
 * Executes platform runtime calibration for a {@link PlatformCalibrationRequest}.
 *
 * <p>Sessions are intended to be used once. The default implementation evaluates
 * each calibration family in order, scores candidates across that family's
 * workloads, promotes the winning runtime profile to the next step, and
 * optionally saves the final profile.</p>
 */
public interface PlatformCalibrationSession {
    /**
     * Runs all calibration steps.
     *
     * @return result with per-step reports, selected profiles, and persistence status
     */
    PlatformCalibrationResult run();

    /**
     * Creates a calibration session with default measurement, validation, and JSON
     * runtime-profile persistence.
     *
     * @param request calibration request
     * @return new session
     */
    static PlatformCalibrationSession create(PlatformCalibrationRequest request) {
        return create(
                request,
                new DefaultMeasurementEngine(),
                new DefaultValidationEngine(),
                new JsonFilePlatformRuntimeProfileStore(request.seedRuntimeProfile())
        );
    }

    /**
     * Creates a calibration session with caller-supplied collaborators.
     *
     * @param request calibration request
     * @param measurementEngine engine used to measure runtime candidates
     * @param validationEngine engine used before measurement
     * @param profileStore store used when the request has an output profile path
     * @return new session
     */
    static PlatformCalibrationSession create(
            PlatformCalibrationRequest request,
            MeasurementEngine measurementEngine,
            ValidationEngine validationEngine,
            PlatformRuntimeProfileStore profileStore
    ) {
        return new DefaultPlatformCalibrationSession(request, measurementEngine, validationEngine, profileStore);
    }
}

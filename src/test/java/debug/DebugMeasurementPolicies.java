package debug;

import tuning.measure.MeasurementPolicy;

final class DebugMeasurementPolicies {
    static final MeasurementPolicy STANDARD = new MeasurementPolicy(
            30,
            100,
            3,
            true,
            true,
            true,
            true,
            false
    );

    static final MeasurementPolicy STANDARD_WITH_TRACE = new MeasurementPolicy(
            30,
            100,
            3,
            true,
            true,
            true,
            true,
            true
    );

    private DebugMeasurementPolicies() {
    }
}

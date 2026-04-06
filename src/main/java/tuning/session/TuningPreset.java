package tuning.session;

import tuning.measure.MeasurementPolicy;
import tuning.report.ReportPolicy;
import tuning.search.SearchPolicy;
import tuning.validate.ValidationPolicy;

public enum TuningPreset {
    QUICK,
    BALANCED,
    THOROUGH;

    public MeasurementPolicy benchmarkMeasurement() {
        return switch (this) {
            case QUICK -> TuningDefaults.quickMeasurement();
            case BALANCED -> TuningDefaults.balancedMeasurement();
            case THOROUGH -> TuningDefaults.thoroughMeasurement();
        };
    }

    public ValidationPolicy benchmarkValidation() {
        return switch (this) {
            case QUICK -> TuningDefaults.quickValidation();
            case BALANCED -> TuningDefaults.balancedValidation();
            case THOROUGH -> TuningDefaults.thoroughValidation();
        };
    }

    public MeasurementPolicy autotuneMeasurement() {
        return benchmarkMeasurement();
    }

    public ValidationPolicy autotuneValidation() {
        return switch (this) {
            case QUICK -> TuningDefaults.quickValidation();
            case BALANCED -> TuningDefaults.balancedValidation();
            case THOROUGH -> TuningDefaults.thoroughValidation();
        };
    }

    public SearchPolicy autotuneSearch() {
        return switch (this) {
            case QUICK -> TuningDefaults.quickSearchPolicy();
            case BALANCED -> TuningDefaults.balancedSearchPolicy();
            case THOROUGH -> TuningDefaults.thoroughSearchPolicy();
        };
    }

    public ReportPolicy reportPolicy() {
        return TuningDefaults.defaultReportPolicy();
    }

    public BaselinePolicy baselinePolicy() {
        return BaselinePolicy.defaults();
    }
}

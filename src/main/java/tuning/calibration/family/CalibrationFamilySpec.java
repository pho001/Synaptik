package tuning.calibration.family;

import tensor.DataType;

import java.util.Set;

public record CalibrationFamilySpec(
        CalibrationFamilyId id,
        String cliName,
        Set<DataType> supportedDTypes,
        boolean acceleratorOptIn,
        Set<String> ownedKnobs
) {
    public CalibrationFamilySpec {
        if (id == null) {
            throw new IllegalArgumentException("id cannot be null");
        }
        if (cliName == null || cliName.isBlank()) {
            throw new IllegalArgumentException("cliName cannot be blank");
        }
        supportedDTypes = supportedDTypes == null ? Set.of() : Set.copyOf(supportedDTypes);
        ownedKnobs = ownedKnobs == null ? Set.of() : Set.copyOf(ownedKnobs);
    }
}

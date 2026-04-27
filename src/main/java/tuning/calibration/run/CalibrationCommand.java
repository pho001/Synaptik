package tuning.calibration.run;

import backend.runtime.ExecutionMode;
import tensor.DataType;
import tuning.calibration.family.CalibrationFamilyId;
import tuning.calibration.family.CalibrationFamilyRegistry;
import tuning.measure.MeasurementPolicy;
import tuning.preset.TuningPreset;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public record CalibrationCommand(
        List<DataType> dataTypes,
        CalibrationFamilyId family,
        CalibrationScope scope,
        TuningPreset preset,
        ExecutionMode mode,
        MeasurementPolicy measurement,
        String colorMode,
        String progressMode,
        Path outputRoot,
        boolean includeAccelerators
) {
    private static final List<DataType> SUPPORTED_DTYPES = List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16);

    public CalibrationCommand {
        dataTypes = dataTypes == null ? List.of() : List.copyOf(dataTypes);
        if (dataTypes.isEmpty()) {
            throw new IllegalArgumentException("At least one calibration dtype is required.");
        }
        for (DataType dataType : dataTypes) {
            if (!SUPPORTED_DTYPES.contains(dataType)) {
                throw new IllegalArgumentException("Unsupported calibration dtype: " + dataType);
            }
        }
        scope = scope == null ? (family == null ? CalibrationScope.ALL_FAMILIES : CalibrationScope.SINGLE_FAMILY) : scope;
        if (scope == CalibrationScope.SINGLE_FAMILY && family == null) {
            throw new IllegalArgumentException("Single-family calibration requires --family.");
        }
        preset = preset == null ? TuningPreset.BALANCED : preset;
        mode = mode == null ? ExecutionMode.FORWARD_BACKWARD : mode;
        colorMode = (colorMode == null || colorMode.isBlank()) ? "auto" : colorMode;
        progressMode = (progressMode == null || progressMode.isBlank()) ? "live" : progressMode;
        outputRoot = outputRoot == null ? Path.of("profiles") : outputRoot;
    }

    public static CalibrationCommand parse(String[] args) {
        if (args == null || args.length == 0 || !"calibrate".equalsIgnoreCase(args[0])) {
            throw new IllegalArgumentException("Calibration command must start with `calibrate`.");
        }
        List<DataType> dataTypes = null;
        CalibrationFamilyId family = null;
        boolean allFamilies = false;
        TuningPreset preset = TuningPreset.BALANCED;
        ExecutionMode mode = ExecutionMode.FORWARD_BACKWARD;
        MeasurementPolicy measurement = null;
        String color = "auto";
        String progress = "live";
        Path outputRoot = Path.of("profiles");
        boolean includeAccelerators = false;

        for (int i = 1; i < args.length; i++) {
            String arg = args[i];
            switch (arg) {
                case "--dtype" -> {
                    requireValue(args, i, arg);
                    if (dataTypes != null) {
                        throw new IllegalArgumentException("Use either --dtype or --dtypes all, not both.");
                    }
                    dataTypes = List.of(parseDType(args[++i]));
                }
                case "--dtypes" -> {
                    requireValue(args, i, arg);
                    if (dataTypes != null) {
                        throw new IllegalArgumentException("Use either --dtype or --dtypes all, not both.");
                    }
                    String value = args[++i];
                    if (!"all".equalsIgnoreCase(value)) {
                        throw new IllegalArgumentException("--dtypes only supports `all`.");
                    }
                    dataTypes = SUPPORTED_DTYPES;
                }
                case "--family" -> {
                    requireValue(args, i, arg);
                    if (allFamilies) {
                        throw new IllegalArgumentException("Use either --family or --families all, not both.");
                    }
                    family = CalibrationFamilyRegistry.parse(args[++i]);
                }
                case "--families" -> {
                    requireValue(args, i, arg);
                    if (family != null) {
                        throw new IllegalArgumentException("Use either --family or --families all, not both.");
                    }
                    String value = args[++i];
                    if (!"all".equalsIgnoreCase(value)) {
                        throw new IllegalArgumentException("--families only supports `all`.");
                    }
                    allFamilies = true;
                }
                case "--preset" -> {
                    requireValue(args, i, arg);
                    preset = TuningPreset.valueOf(args[++i].trim().toUpperCase(Locale.ROOT));
                }
                case "--mode" -> {
                    requireValue(args, i, arg);
                    mode = parseMode(args[++i]);
                }
                case "--measurement" -> {
                    requireValue(args, i, arg);
                    measurement = parseMeasurement(args[++i], preset);
                }
                case "--color" -> {
                    requireValue(args, i, arg);
                    color = args[++i].trim().toLowerCase(Locale.ROOT);
                    if (!List.of("auto", "always", "never").contains(color)) {
                        throw new IllegalArgumentException("--color must be auto, always, or never.");
                    }
                }
                case "--progress" -> {
                    requireValue(args, i, arg);
                    progress = args[++i].trim().toLowerCase(Locale.ROOT);
                    if (!List.of("live", "lines", "quiet").contains(progress)) {
                        throw new IllegalArgumentException("--progress must be live, lines, or quiet.");
                    }
                }
                case "--output-root" -> {
                    requireValue(args, i, arg);
                    outputRoot = Path.of(args[++i]);
                }
                case "--include-accelerators" -> includeAccelerators = true;
                default -> throw new IllegalArgumentException("Unknown calibration option: " + arg);
            }
        }

        if (dataTypes == null) {
            throw new IllegalArgumentException("Calibration requires --dtype <f64|f32|bf16> or --dtypes all.");
        }
        CalibrationScope scope = allFamilies ? CalibrationScope.ALL_FAMILIES : CalibrationScope.SINGLE_FAMILY;
        if (!allFamilies && family == null) {
            throw new IllegalArgumentException("Calibration requires --family <family-id> or --families all.");
        }
        return new CalibrationCommand(dataTypes, family, scope, preset, mode, measurement, color, progress, outputRoot, includeAccelerators);
    }

    public int passCount() {
        if (scope == CalibrationScope.SINGLE_FAMILY || preset == TuningPreset.QUICK) {
            return 1;
        }
        return 2;
    }

    public List<CalibrationFamilyId> families() {
        if (scope == CalibrationScope.SINGLE_FAMILY) {
            return List.of(family);
        }
        return CalibrationFamilyRegistry.fullSuite(includeAccelerators);
    }

    private static void requireValue(String[] args, int index, String option) {
        if (index + 1 >= args.length) {
            throw new IllegalArgumentException(option + " requires a value.");
        }
    }

    private static DataType parseDType(String value) {
        return switch (value.trim().toLowerCase(Locale.ROOT)) {
            case "f64" -> DataType.FLOAT64;
            case "f32" -> DataType.FLOAT32;
            case "bf16" -> DataType.BFLOAT16;
            case "i32", "int32", "bool" -> throw new IllegalArgumentException("Calibration supports only f64, f32, and bf16.");
            default -> throw new IllegalArgumentException("Unknown calibration dtype: " + value);
        };
    }

    private static ExecutionMode parseMode(String value) {
        return switch (value.trim().toLowerCase(Locale.ROOT)) {
            case "forward" -> ExecutionMode.FORWARD;
            case "forward-backward", "forward_backward", "training" -> ExecutionMode.FORWARD_BACKWARD;
            default -> throw new IllegalArgumentException("Unsupported calibration mode: " + value);
        };
    }

    private static MeasurementPolicy parseMeasurement(String value, TuningPreset preset) {
        String[] parts = value.split(":");
        if (parts.length != 3) {
            throw new IllegalArgumentException("--measurement expects warmup:measure:repeats.");
        }
        try {
            MeasurementPolicy base = preset.benchmarkMeasurement();
            return new MeasurementPolicy(
                    Integer.parseInt(parts[0]),
                    Integer.parseInt(parts[1]),
                    Integer.parseInt(parts[2]),
                    base.measureCompile(),
                    base.measurePrepare(),
                    base.measureColdRun(),
                    base.measureSteadyState(),
                    base.captureStepTrace()
            );
        } catch (NumberFormatException ex) {
            throw new IllegalArgumentException("--measurement values must be integers.", ex);
        }
    }

    public String describePlan() {
        List<String> dtypeNames = new ArrayList<>();
        for (DataType dataType : dataTypes) {
            dtypeNames.add(dtypeId(dataType));
        }
        return "Calibration plan: dtypes=" + dtypeNames
                + ", families=" + (scope == CalibrationScope.ALL_FAMILIES ? "all" : CalibrationFamilyRegistry.spec(family).cliName())
                + ", preset=" + preset.name().toLowerCase(Locale.ROOT)
                + ", mode=" + mode.name().toLowerCase(Locale.ROOT)
                + ", passes=" + passCount()
                + ", progress=" + progressMode
                + ", color=" + colorMode;
    }

    public static String dtypeId(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> "f64";
            case FLOAT32 -> "f32";
            case BFLOAT16 -> "bf16";
            case INT32 -> "i32";
            case BOOL -> "bool";
        };
    }
}

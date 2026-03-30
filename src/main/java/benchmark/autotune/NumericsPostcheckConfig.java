package benchmark.autotune;

import tensor.DataType;

import java.nio.file.Path;
import java.time.format.DateTimeFormatter;

public record NumericsPostcheckConfig(
        DataType dtype,
        int topN,
        Path reportDir,
        DateTimeFormatter timestampFormat
) {}

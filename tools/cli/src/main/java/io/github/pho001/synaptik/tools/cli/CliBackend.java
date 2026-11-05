package io.github.pho001.synaptik.tools.cli;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.engine.Engine;
import java.nio.file.Path;

/** Explicit public Engine composition, with no alternate-backend retries or selection claims. */
record CliBackend(String name, Path library, CpuBackendIntegration.OneDnnApi api) {
    static CliBackend parse(CliArguments args) {
        String name = args.value("backend", "cpu");
        if (!name.equals("cpu") && !name.equals("metal") && !name.equals("onednn")) {
            throw new IllegalArgumentException("--backend must be cpu, metal or onednn");
        }
        if (args.has("metal-library") && !name.equals("metal")) {
            throw new IllegalArgumentException("--metal-library requires --backend metal");
        }
        if ((args.has("onednn-library") || args.has("onednn-api")) && !name.equals("onednn")) {
            throw new IllegalArgumentException("oneDNN options require --backend onednn");
        }
        Path library = null;
        if (!name.equals("cpu")) {
            String option = name.equals("metal") ? "metal-library" : "onednn-library";
            if (!args.has(option)) throw new IllegalArgumentException("required option: --" + option);
            library = Path.of(args.value(option, ""));
            if (!library.isAbsolute()) throw new IllegalArgumentException("--" + option + " must be absolute");
            library = library.normalize();
        }
        var api = switch (args.value("onednn-api", "auto")) {
            case "auto" -> CpuBackendIntegration.OneDnnApi.AUTO;
            case "primitives" -> CpuBackendIntegration.OneDnnApi.PRIMITIVES;
            case "graph" -> CpuBackendIntegration.OneDnnApi.GRAPH;
            default -> throw new IllegalArgumentException("--onednn-api must be auto, primitives or graph");
        };
        return new CliBackend(name, library, api);
    }

    /**
     * Opens an owned Engine; the caller closes Models borrowing it before closing the Engine.
     * @return new Engine with exactly the explicitly selected public composition
     */
    Engine open() {
        if (name.equals("cpu")) return Engine.standard();
        try (var builder = Engine.builder()) {
            if (name.equals("metal")) {
                builder.takeOwnership(MetalBackendIntegration.open(new MetalBackendConfiguration(library)));
            } else {
                builder.takeOwnership(CpuBackendIntegration.openOneDnn(library, api));
            }
            return builder.build();
        }
    }

    String label() {
        return switch (name) {
            case "metal" -> "Metal-only composition";
            case "onednn" -> "CPU / explicit oneDNN " + api + " preference (selection not asserted)";
            default -> "CPU composition";
        };
    }
}

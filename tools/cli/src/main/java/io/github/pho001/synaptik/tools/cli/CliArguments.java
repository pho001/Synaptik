package io.github.pho001.synaptik.tools.cli;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/** Strict command-local options: unknown, duplicate and inapplicable options are errors. */
final class CliArguments {
    private static final Set<String> BACKEND = Set.of(
            "backend", "metal-library", "onednn-library", "onednn-api");
    private final String command;
    private final Map<String, String> values;

    private CliArguments(String command, Map<String, String> values) {
        this.command = command;
        this.values = Map.copyOf(values);
    }

    /**
     * Parses options without accessing files or opening execution owners.
     * @param args command followed by long options; no positional data arguments
     * @return validated syntax with command-specific option names
     * @throws IllegalArgumentException for invalid syntax or an unknown command/option
     */
    static CliArguments parse(String[] args) {
        if (args.length == 0) return new CliArguments("help", Map.of());
        String command = args[0];
        if (command.equals("--help") || command.equals("-h") || command.equals("help")) {
            if (args.length > 1 && !(args.length == 2
                    && Set.of("train", "predict", "inspect").contains(args[1]))) {
                throw new IllegalArgumentException("help accepts only an optional command name");
            }
            return new CliArguments("help", Map.of());
        }
        Set<String> allowed = switch (command) {
            case "train" -> Set.of("data", "output", "model", "epochs", "batch-size", "outputs",
                    "seed", "optimizer", "learning-rate", "refresh-ms", "plain", "help");
            case "predict" -> Set.of("data", "output", "model", "help");
            case "inspect" -> Set.of("model", "help");
            default -> throw new IllegalArgumentException("unknown command: " + command);
        };
        Map<String, String> values = new HashMap<>();
        for (int i = 1; i < args.length; i++) {
            String token = args[i];
            if (!token.startsWith("--") || token.length() == 2) {
                throw new IllegalArgumentException("expected a long option: " + token);
            }
            int equal = token.indexOf('=');
            String name = token.substring(2, equal < 0 ? token.length() : equal);
            if (!allowed.contains(name) && !(BACKEND.contains(name) && !command.equals("inspect"))) {
                throw new IllegalArgumentException("unsupported option for " + command + ": --" + name);
            }
            boolean flag = name.equals("plain") || name.equals("help");
            String value;
            if (flag) {
                if (equal >= 0) throw new IllegalArgumentException("--" + name + " takes no value");
                value = "true";
            } else if (equal >= 0) {
                value = token.substring(equal + 1);
            } else {
                if (++i == args.length || args[i].startsWith("--")) {
                    throw new IllegalArgumentException("missing value for --" + name);
                }
                value = args[i];
            }
            if (value.isBlank()) throw new IllegalArgumentException("empty value for --" + name);
            if (values.putIfAbsent(name, value) != null) {
                throw new IllegalArgumentException("duplicate option: --" + name);
            }
        }
        return new CliArguments(values.containsKey("help") ? "help" : command, values);
    }

    String command() { return command; }
    boolean has(String name) { return values.containsKey(name); }
    String value(String name, String fallback) { return values.getOrDefault(name, fallback); }

    Path requiredPath(String name) {
        if (!has(name)) throw new IllegalArgumentException("required option: --" + name);
        return Path.of(values.get(name)).toAbsolutePath().normalize();
    }

    long integer(String name, long fallback) {
        if (!has(name)) return fallback;
        try {
            return Long.parseLong(values.get(name));
        } catch (NumberFormatException failure) {
            throw new IllegalArgumentException("--" + name + " requires an integer", failure);
        }
    }

    int positiveInt(String name, int fallback) {
        long value = integer(name, fallback);
        if (value <= 0 || value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("--" + name + " must be in 1.." + Integer.MAX_VALUE);
        }
        return (int) value;
    }

    double positiveFloatRate(String name, double fallback) {
        double value;
        try {
            value = Double.parseDouble(value(name, Double.toString(fallback)));
        } catch (NumberFormatException failure) {
            throw new IllegalArgumentException("--" + name + " requires a number", failure);
        }
        if (!Double.isFinite(value) || !Float.isFinite((float) value) || (float) value <= 0) {
            throw new IllegalArgumentException("--" + name + " must be finite and positive in FLOAT32");
        }
        return value;
    }
}

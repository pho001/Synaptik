package io.github.pho001.synaptik.tools.cli;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * UI-thread form state. Values become ordinary shared workflow options; numerical/file validation
 * remains in the common parser/workflow. Disabled controls are visible but never emitted.
 */
final class TuiForm {
    enum Action { TRAIN, PREDICT, INSPECT }
    enum Result { CHANGED, SUBMIT, BACK, UNCHANGED }
    record Field(String label, String value, boolean enabled, boolean choice) {}

    private static final class Input {
        final String option;
        final String label;
        final List<String> choices;
        String value;

        Input(String option, String label, String value, String... choices) {
            this.option = option;
            this.label = label;
            this.value = value;
            this.choices = List.of(choices);
        }
    }

    private final Action action;
    private final List<Input> inputs = new ArrayList<>();
    private int focused;
    private int cursor;

    /**
     * Creates a fresh editable form using the shared CLI defaults.
     * @param action command to collect; nonnull
     */
    TuiForm(Action action) {
        this.action = java.util.Objects.requireNonNull(action);
        if (action == Action.INSPECT) {
            add("model", "Model file (required)", "");
        } else {
            add("data", "CSV data (required)", "");
            add("output", action == Action.TRAIN ? "Save model (required)" : "Output CSV (required)", "");
            add("model", action == Action.TRAIN ? "Load model (optional)" : "Model file (required)", "");
            if (action == Action.TRAIN) {
                add("epochs", "Epochs", Integer.toString(CliDefaults.EPOCHS));
                add("batch-size", "Batch size (new model)", Integer.toString(CliDefaults.BATCH_SIZE));
                add("outputs", "Outputs (new model)", Integer.toString(CliDefaults.OUTPUTS));
                add("seed", "Seed (new model)", Long.toString(CliDefaults.SEED));
                add("optimizer", "Optimizer", CliDefaults.OPTIMIZER, "sgd", "adam");
                add("learning-rate", "Learning rate (blank = optimizer default)", "");
            }
            add("backend", "Backend", CliDefaults.BACKEND, "cpu", "metal", "onednn");
            add("metal-library", "Metal library (absolute path)", "");
            add("onednn-library", "oneDNN library (absolute path)", "");
            add("onednn-api", "oneDNN API preference", CliDefaults.ONEDNN_API, "auto", "primitives", "graph");
            if (action == Action.TRAIN) add("refresh-ms", "Refresh milliseconds", Long.toString(CliDefaults.REFRESH_MILLIS));
        }
        cursor = inputs.getFirst().value.length();
    }

    private void add(String option, String label, String value, String... choices) {
        inputs.add(new Input(option, label, value, choices));
    }

    /** @return the shared workflow selected for this form */
    Action action() { return action; }

    /** @return zero-based focused field index, including visible disabled fields */
    int focused() { return focused; }

    /** @return UTF16 offset at a code-point boundary within the focused field */
    int cursorOffset() { return cursor; }

    /** @return detached visible fields with current enablement; disabled values are retained for editing later */
    List<Field> fields() {
        List<Field> result = new ArrayList<>(inputs.size());
        for (Input input : inputs) result.add(new Field(input.label, input.value, enabled(input), !input.choices.isEmpty()));
        return List.copyOf(result);
    }

    private String value(String option) {
        for (Input input : inputs) if (input.option.equals(option)) return input.value;
        return "";
    }

    private boolean enabled(Input input) {
        return switch (input.option) {
            case "batch-size", "outputs", "seed" -> value("model").isBlank();
            case "metal-library" -> value("backend").equals("metal");
            case "onednn-library", "onednn-api" -> value("backend").equals("onednn");
            default -> true;
        };
    }

    /**
     * Edits one code point or moves focus; Enter starts and Escape returns without modifying data.
     * @param key decoded nonnull terminal event
     * @return UI transition, without starting a worker or opening any file
     */
    Result handle(TuiConsole.Key key) {
        switch (key.kind()) {
            case ENTER: return Result.SUBMIT;
            case ESCAPE: return Result.BACK;
            case TAB, DOWN: move(1); return Result.CHANGED;
            case BACK_TAB, UP: move(-1); return Result.CHANGED;
            default: break;
        }
        Input input = inputs.get(focused);
        if (!enabled(input)) return Result.UNCHANGED;
        if (!input.choices.isEmpty()) return choice(input, key);
        String value = input.value;
        switch (key.kind()) {
            case LEFT:
                if (cursor == 0) return Result.UNCHANGED;
                cursor = value.offsetByCodePoints(cursor, -1);
                break;
            case RIGHT:
                if (cursor == value.length()) return Result.UNCHANGED;
                cursor = value.offsetByCodePoints(cursor, 1);
                break;
            case HOME: cursor = 0; break;
            case END: cursor = value.length(); break;
            case BACKSPACE:
                if (cursor == 0) return Result.UNCHANGED;
                int start = value.offsetByCodePoints(cursor, -1);
                input.value = value.substring(0, start) + value.substring(cursor);
                cursor = start;
                break;
            case DELETE:
                if (cursor == value.length()) return Result.UNCHANGED;
                input.value = value.substring(0, cursor) + value.substring(value.offsetByCodePoints(cursor, 1));
                break;
            case CHARACTER:
                int cp = key.character();
                if (!Character.isValidCodePoint(cp) || cp >= 0xd800 && cp <= 0xdfff
                        || Character.isISOControl(cp)) return Result.UNCHANGED;
                String character = Character.toString(cp);
                input.value = value.substring(0, cursor) + character + value.substring(cursor);
                cursor += character.length();
                break;
            default: return Result.UNCHANGED;
        }
        return Result.CHANGED;
    }

    private Result choice(Input input, TuiConsole.Key key) {
        int index = input.choices.indexOf(input.value);
        int next;
        switch (key.kind()) {
            case LEFT: next = Math.floorMod(index - 1, input.choices.size()); break;
            case RIGHT: next = (index + 1) % input.choices.size(); break;
            case HOME: next = 0; break;
            case END: next = input.choices.size() - 1; break;
            case CHARACTER:
                if (key.character() == ' ') next = (index + 1) % input.choices.size();
                else {
                    next = index;
                    for (int i = 1; i <= input.choices.size(); i++) {
                        int candidate = (index + i) % input.choices.size();
                        if (input.choices.get(candidate).codePointAt(0) == Character.toLowerCase(key.character())) {
                            next = candidate;
                            break;
                        }
                    }
                }
                break;
            default: return Result.UNCHANGED;
        }
        if (next == index) return Result.UNCHANGED;
        input.value = input.choices.get(next);
        cursor = input.value.length();
        return Result.CHANGED;
    }

    private void move(int direction) {
        do {
            focused = Math.floorMod(focused + direction, inputs.size());
        } while (!enabled(inputs.get(focused)));
        cursor = inputs.get(focused).value.length();
    }

    /**
     * Produces the exact shared command-line option representation without a second validator.
     * TUI alone requires prediction output so shared prediction cannot print CSV onto the screen.
     * @return detached command and enabled, nonblank option/value pairs
     * @throws IllegalArgumentException if the TUI-required prediction output is blank
     */
    String[] arguments() {
        if (action == Action.PREDICT && value("output").isBlank()) {
            throw new IllegalArgumentException("TUI prediction requires an output CSV file");
        }
        List<String> args = new ArrayList<>();
        args.add(action.name().toLowerCase(Locale.ROOT));
        for (Input input : inputs) {
            if (enabled(input) && !input.value.isBlank()) {
                // Equals form preserves paths beginning with '--' as ordinary option values.
                args.add("--" + input.option + "=" + input.value);
            }
        }
        return args.toArray(String[]::new);
    }
}

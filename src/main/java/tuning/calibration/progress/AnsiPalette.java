package tuning.calibration.progress;

public final class AnsiPalette {
    public static final String RESET = "\u001B[0m";
    public static final String BOLD = "\u001B[1m";
    public static final String CYAN = "\u001B[36m";
    public static final String BLUE = "\u001B[34m";
    public static final String GREEN = "\u001B[32m";
    public static final String YELLOW = "\u001B[33m";
    public static final String CLEAR_LINE = "\u001B[2K";

    private AnsiPalette() {
    }

    public static String color(String value, String code, boolean enabled) {
        if (!enabled || value == null || value.isEmpty()) {
            return value == null ? "" : value;
        }
        return code + value + RESET;
    }
}

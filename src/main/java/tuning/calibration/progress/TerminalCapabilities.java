package tuning.calibration.progress;

import java.io.PrintStream;

public record TerminalCapabilities(boolean colorEnabled, boolean liveRedrawEnabled) {
    public static TerminalCapabilities detect(String colorMode, String progressMode, PrintStream out) {
        String safeColor = colorMode == null ? "auto" : colorMode;
        String safeProgress = progressMode == null ? "live" : progressMode;
        boolean interactive = System.console() != null;
        boolean color = switch (safeColor) {
            case "always" -> true;
            case "never" -> false;
            default -> interactive;
        };
        boolean live = "live".equals(safeProgress) && interactive && out != null;
        return new TerminalCapabilities(color, live);
    }
}

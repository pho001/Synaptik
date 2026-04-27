package tuning.calibration.progress;

public final class SynaptikBanner {
    private static final String TEXT = """
              _____                         _   _ _
             / ____|                       | | (_) |
            | (___  _   _ _ __   __ _ _ __ | |_ _| | __
             \\___ \\| | | | '_ \\ / _` | '_ \\| __| | |/ /
             ____) | |_| | | | | (_| | |_) | |_| |   <
            |_____/ \\__, |_| |_|\\__,_| .__/ \\__|_|_|\\_\\
                     __/ |           | |
                    |___/            |_|
            """;

    private SynaptikBanner() {
    }

    public static String render(boolean colorEnabled) {
        return AnsiPalette.color(TEXT, AnsiPalette.CYAN + AnsiPalette.BOLD, colorEnabled);
    }
}

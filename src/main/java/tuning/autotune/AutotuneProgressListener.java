package tuning.autotune;

public interface AutotuneProgressListener {
    void onEvent(AutotuneProgressEvent event);

    static AutotuneProgressListener noop() {
        return event -> { };
    }
}

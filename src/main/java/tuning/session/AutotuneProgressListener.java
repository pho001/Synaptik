package tuning.session;

public interface AutotuneProgressListener {
    void onEvent(AutotuneProgressEvent event);

    static AutotuneProgressListener noop() {
        return event -> { };
    }
}

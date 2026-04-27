package backend.lowering;

import java.util.List;

public record LoweringTrace(
        List<String> events
) {
    public LoweringTrace {
        events = List.copyOf(events == null ? List.of() : events);
    }

    public static LoweringTrace empty() {
        return new LoweringTrace(List.of());
    }
}

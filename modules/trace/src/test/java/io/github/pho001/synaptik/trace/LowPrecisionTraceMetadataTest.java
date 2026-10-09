package io.github.pho001.synaptik.trace;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.trace.payload.LowPrecisionTraceMetadata;
import io.github.pho001.synaptik.trace.payload.TraceRouteKind;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

final class LowPrecisionTraceMetadataTest {
    @Test
    void snapshotsCustomRouteExecutionFacts() {
        var tuple = new ArrayList<>(List.of("BFLOAT16", "FLOAT16"));
        var metadata = new LowPrecisionTraceMetadata(
                TraceRouteKind.CUSTOM_KERNEL,
                tuple);
        tuple.set(0, "FLOAT32");

        assertEquals(List.of("BFLOAT16", "FLOAT16"), metadata.logicalDtypeTuple());
        assertEquals(TraceRouteKind.CUSTOM_KERNEL, metadata.selectedRoute());
        assertEquals(List.of("selectedRoute", "logicalDtypeTuple"),
                Arrays.stream(LowPrecisionTraceMetadata.class.getRecordComponents())
                        .map(component -> component.getName()).toList());
        assertThrows(UnsupportedOperationException.class,
                () -> metadata.logicalDtypeTuple().add("FLOAT16"));
    }

    @Test
    void rawPreservingRouteUsesTheSameExecutionFacts() {
        var metadata = new LowPrecisionTraceMetadata(
                TraceRouteKind.CUSTOM_KERNEL,
                List.of("FLOAT16", "FLOAT16"));

        assertEquals(TraceRouteKind.CUSTOM_KERNEL, metadata.selectedRoute());
        assertEquals(List.of("FLOAT16", "FLOAT16"), metadata.logicalDtypeTuple());
    }

    @Test
    void rejectsGraphRoutesAndMalformedExecutionFacts() {
        assertThrows(IllegalArgumentException.class, () -> new LowPrecisionTraceMetadata(
                TraceRouteKind.GRAPH_EXECUTABLE,
                List.of("FLOAT16")));
        assertThrows(IllegalArgumentException.class, () -> new LowPrecisionTraceMetadata(
                TraceRouteKind.CUSTOM_KERNEL,
                List.of()));
        assertThrows(IllegalArgumentException.class, () -> new LowPrecisionTraceMetadata(
                TraceRouteKind.CUSTOM_KERNEL,
                List.of("")));
    }
}

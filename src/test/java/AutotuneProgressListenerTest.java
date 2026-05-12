import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tuning.candidate.Candidate;
import tuning.candidate.ListCandidateSpace;
import tuning.autotune.AutotuneProgressEvent;
import tuning.autotune.AutotuneProgressListener;
import tuning.autotune.AutotuneProgressPhase;
import tuning.autotune.AutotuneRequest;
import tuning.autotune.AutotuneSession;
import tuning.validate.TensorSnapshot;
import tuning.validate.ValidationPolicy;
import tuning.validate.ValidationReference;
import tuning.workload.TensorRootWorkloadSpec;
import tuning.workload.WorkloadKind;
import tuning.workload.WorkloadMetadata;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class AutotuneProgressListenerTest {
    @Test
    void autotuneEmitsProgressEventsAcrossLifecycle() {
        TensorRootWorkloadSpec workload = new TensorRootWorkloadSpec(
                "autotune_progress",
                WorkloadKind.GENERIC,
                environment -> Tensor.scalar(2.0).add(Tensor.scalar(3.0)),
                environment -> ValidationReference.snapshot(
                        TensorSnapshot.capture("out", Tensor.scalar(5.0)),
                        Map.of(),
                        List.of()
                ),
                environment -> WorkloadMetadata.of("autotune_progress", WorkloadKind.GENERIC)
        );

        List<AutotuneProgressEvent> events = new ArrayList<>();
        AutotuneProgressListener listener = events::add;

        Candidate candidate = new Candidate("valid", profile("valid"));
        AutotuneSession.create(new AutotuneRequest(
                workload,
                new ListCandidateSpace(List.of(candidate)),
                new tuning.measure.MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                ValidationPolicy.defaults(),
                new tuning.search.SearchPolicy(10, 1, 1, false),
                tuning.store.PersistencePolicy.disabled(),
                listener
        )).run();

        assertTrue(events.stream().anyMatch(event -> event.phase() == AutotuneProgressPhase.STARTED));
        assertTrue(events.stream().anyMatch(event -> event.phase() == AutotuneProgressPhase.SEARCH_BATCH));
        assertTrue(events.stream().anyMatch(event -> event.phase() == AutotuneProgressPhase.CANDIDATE_VALIDATING));
        assertTrue(events.stream().anyMatch(event -> event.phase() == AutotuneProgressPhase.CANDIDATE_MEASURED));
        assertTrue(events.stream().anyMatch(event -> event.phase() == AutotuneProgressPhase.COMPLETED));
    }

    @Test
    void autotuneEmitsInvalidCandidateEvents() {
        TensorRootWorkloadSpec workload = new TensorRootWorkloadSpec(
                "autotune_progress_invalid",
                WorkloadKind.GENERIC,
                environment -> Tensor.scalar(2.0).add(Tensor.scalar(3.0)),
                environment -> ValidationReference.snapshot(
                        TensorSnapshot.capture("out", Tensor.scalar(99.0)),
                        Map.of(),
                        List.of()
                ),
                environment -> WorkloadMetadata.of("autotune_progress_invalid", WorkloadKind.GENERIC)
        );

        List<AutotuneProgressEvent> events = new ArrayList<>();
        Candidate candidate = new Candidate("invalid", profile("invalid"));
        AutotuneSession.create(new AutotuneRequest(
                workload,
                new ListCandidateSpace(List.of(candidate)),
                new tuning.measure.MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                ValidationPolicy.defaults(),
                new tuning.search.SearchPolicy(10, 1, 1, false),
                tuning.store.PersistencePolicy.disabled(),
                events::add
        )).run();

        assertTrue(events.stream().anyMatch(event -> event.phase() == AutotuneProgressPhase.CANDIDATE_INVALID));
    }

    private static ExecutionProfile profile(String name) {
        return new ExecutionProfile(
                name,
                name,
                DataType.FLOAT64,
                ExecutionMode.FORWARD,
                config.compile.CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
    }
}

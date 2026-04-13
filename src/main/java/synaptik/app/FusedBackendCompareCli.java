package synaptik.app;

import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import config.profile.ExecutionProfile;
import config.profile.PlatformRuntimeProfile;
import config.profile.PlatformRuntimeProfileIO;
import config.profile.WorkloadProfile;
import config.runtime.FusedExecutionPolicy;
import config.runtime.FusedPrimaryBackend;
import config.runtime.RuntimeConfig;
import tensor.DataType;
import tuning.report.TextBenchmarkReportRenderer;
import tuning.session.BenchmarkEntry;
import tuning.session.BenchmarkSession;
import tuning.session.TuningDefaults;
import tuning.session.TuningPreset;
import tuning.workload.StandardWorkloads;

import java.nio.file.Path;
import java.util.List;

public final class FusedBackendCompareCli {
    private FusedBackendCompareCli() {
    }

    public static void main(String[] args) {
        ExecutionProfile seed = fusedSeed();
        ExecutionProfile direct = withBackend(seed, "direct-vector", FusedPrimaryBackend.DIRECT_VECTOR);
        ExecutionProfile asm = withBackend(seed, "asm", FusedPrimaryBackend.ASM);

        var request = TuningDefaults.benchmark(
                TuningPreset.BALANCED,
                StandardWorkloads.abcSequenceMatmul("abc_sequence_matmul_f64", 64, 10_000),
                List.of(
                        BenchmarkEntry.candidate("direct-vector", direct),
                        BenchmarkEntry.candidate("asm", asm)
                )
        );
        var report = BenchmarkSession.create(request).run();
        System.out.println(TextBenchmarkReportRenderer.render(report));
    }

    private static ExecutionProfile fusedSeed() {
        ExecutionProfile fallback = new ExecutionProfile(
                "platform-seed-f64-training",
                "platform-seed-f64-training",
                DataType.FLOAT64,
                ExecutionMode.FORWARD_BACKWARD,
                OptimizerConfig.trainingDefaults(),
                RuntimeConfig.trainingDefaults(),
                WorkloadProfile.none()
        );
        PlatformRuntimeProfile runtime = PlatformRuntimeProfileIO.loadOrDefault(
                Path.of("build/platform-calibration/f64/profiles/platform/mac_os_x-aarch64-oracle_corporation-16c/f64-forward-backward.json"),
                PlatformRuntimeProfile.fromExecutionProfile("fallback", "fallback", "fallback", fallback)
        );
        return new ExecutionProfile(
                "abc-f64-fused-compare",
                "abc-f64-fused-compare",
                DataType.FLOAT64,
                ExecutionMode.FORWARD_BACKWARD,
                new OptimizerConfig(
                        List.of(OptimizerStage.FUSE, OptimizerStage.CSE, OptimizerStage.MEM),
                        OptimizerConfig.trainingDefaults().rewrite(),
                        OptimizerConfig.trainingDefaults().cse(),
                        OptimizerConfig.trainingDefaults().fuse(),
                        OptimizerConfig.trainingDefaults().memory()
                ),
                runtime.toRuntimeConfig(),
                WorkloadProfile.none()
        );
    }

    private static ExecutionProfile withBackend(ExecutionProfile base, String candidateName, FusedPrimaryBackend backend) {
        return new ExecutionProfile(
                base.profileName(),
                candidateName,
                base.dataType(),
                base.mode(),
                base.optimizer(),
                new RuntimeConfig(
                        base.runtime().kernel(),
                        base.runtime().approximation(),
                        base.runtime().blas(),
                        new FusedExecutionPolicy(backend, true)
                ),
                base.workload()
        );
    }
}

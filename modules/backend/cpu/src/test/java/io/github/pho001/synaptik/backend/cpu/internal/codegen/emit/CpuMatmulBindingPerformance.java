package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.*;
import java.util.concurrent.locks.LockSupport;

/**
 * Explicit bounded observation of the actual joint prepared forward/AD invocation and independent
 * direct Java mathematics. No JUnit or legacy timing selector launches this class. MAIN must
 * separately authorize launch after source freeze and named gates; the argument prevents
 * accidental invocation, not permission. This is neither phase-ABI nor pure-loop timing, and
 * makes no historical byte/resource or performance parity claim.
 */
public final class CpuMatmulBindingPerformance {
    /** Prevents construction of the explicit protocol entry. */
    private CpuMatmulBindingPerformance() {}

    /**
     * Runs four dtype/length cases under a hard 60-second JVM deadline. Production timing includes
     * the complete bound joint invocation; direct timing includes both Java forward and reverse.
     * Compilation and production preparation/binding are reported separately. No old artifact,
     * reconstructed phase entry, child process or retry participates.
     * @param args exactly one literal --main-prelaunch-go, only after actual MAIN authorization
     * @throws Throwable if a semantic check or deadline fails; no retry is allowed
     */
    public static void main(String[] args) throws Throwable {
        if (!Arrays.equals(args,new String[]{"--main-prelaunch-go"}))
            throw new IllegalArgumentException("explicit MAIN prelaunch GO required; never a normal test selector");
        long deadline = System.nanoTime()+60_000_000_000L;
        Thread watchdog = new Thread(() -> {
            while (System.nanoTime() < deadline) LockSupport.parkNanos(deadline-System.nanoTime());
            System.err.println("CPU0071 hard 60-second JVM deadline; no retry"); Runtime.getRuntime().halt(124);
        },"cpu0071-hard-deadline");
        watchdog.setDaemon(true); watchdog.start();
        System.out.println("Joint prepared forward/AD versus direct Java; T4/B7/F3/H2; 10000 warmup/arm/case; 15 alternating pairs x 1000 calls; no retry or historical parity claim");
        System.out.println("case,production_joint_ns_per_call,direct_joint_ns_per_call,production_over_direct");
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (boolean length : List.of(false,true))
            try (var fixture = new CpuMatmulBindingFixture(type,length)) {
                System.out.println("cold,"+type+","+length+",compile_ns="+fixture.compileNanos
                        +",production_prepare_bind_ns="+fixture.prepareBindNanos+",Engine_admission=not_measured");
                fixture.verify();
                Runnable[] arms = {fixture.production,fixture.direct};
                for (Runnable arm : arms) repeat(arm,10000);
                fixture.verifyLatest();
                double[][] samples = new double[2][15];
                for (int pair = 0; pair < 15; pair++) for (int position = 0; position < 2; position++) {
                    int arm = (pair+position)%2; long started = System.nanoTime();
                    repeat(arms[arm],1000); samples[arm][pair] = (System.nanoTime()-started)/1000.0;
                }
                fixture.verifyLatest();
                for (int pair = 0; pair < 15; pair++) System.out.println("sample,"+type+","+length+","+pair+","+samples[0][pair]+","+samples[1][pair]);
                for (double[] sample : samples) Arrays.sort(sample);
                double production = samples[0][7], direct = samples[1][7];
                System.out.println(type+"-"+(length ? "static-length" : "static-all-valid")
                        +","+production+","+direct+","+production/direct);
            }
        System.out.println("Completed four joint cases; all before/after numerical checks passed; no further fork authorized");
    }

    /**
     * Repeats one already-bound production invocation or independent joint Java control.
     * @param action bound production or direct Java action
     * @param count exact positive call count for this observation
     */
    private static void repeat(Runnable action,int count) {
        for (int call = 0; call < count; call++) action.run();
    }
}

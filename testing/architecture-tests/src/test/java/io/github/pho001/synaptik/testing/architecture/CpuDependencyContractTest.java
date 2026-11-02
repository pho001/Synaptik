package io.github.pho001.synaptik.testing.architecture;

import static org.junit.jupiter.api.Assertions.assertFalse;

import java.util.Objects;
import org.junit.jupiter.api.Test;

/** Guards forbidden production edges using Gradle's declared model, not copied build text. */
final class CpuDependencyContractTest {
    /** Checks direct production edges without prescribing API visibility, order or other dependencies. */
    @Test
    void cpuProductionDependenciesExcludeCompilerAndEngine() {
        String projects = Objects.requireNonNull(
                System.getProperty("synaptik.architecture.cpuProductionProjects"),
                "Gradle must supply the actual CPU production dependency model");
        var dependencies = projects.lines().toList();
        assertFalse(dependencies.contains(":modules:compiler"));
        assertFalse(dependencies.contains(":modules:engine"));
    }
}

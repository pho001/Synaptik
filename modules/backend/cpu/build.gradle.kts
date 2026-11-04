import org.gradle.api.tasks.compile.JavaCompile
import org.gradle.api.tasks.javadoc.Javadoc
import org.gradle.external.javadoc.StandardJavadocDocletOptions
import org.gradle.external.javadoc.JavadocMemberLevel
import org.gradle.api.tasks.testing.Test

dependencies {
    testImplementation(project(":modules:compiler"))
    api(project(":modules:model"))
    implementation(project(":modules:config"))
    api(project(":modules:planning"))
    api(project(":modules:runtime"))
    api(project(":modules:prepare"))
    api(project(":modules:backend-contract"))
    api(project(":modules:trace"))
    implementation(project(":backends:openblas-provider"))
}

tasks.withType<JavaCompile>().configureEach {
    options.compilerArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
}

tasks.withType<Test>().configureEach {
    jvmArgs("--add-modules", "jdk.incubator.vector")
    listOf(
        "synaptik.cpu.loss.structuralEvidenceRoot",
        "synaptik.cpu.loss.performanceEvidenceRoot",
        "synaptik.cpu.loss.performance",
        "synaptik.cpu.loss.performanceRepresentative",
        "synaptik.cpu.loss.c2Evidence",
        "synaptik.cpu.cast.structuralEvidenceRoot",
        "synaptik.cpu.cast.performance",
        "synaptik.cpu.cast.performanceEvidenceRoot",
        "synaptik.cpu.concatStack.performance",
        "synaptik.cpu.concatStack.performanceEvidenceRoot",
        "synaptik.cpu.pointwiseMask.structuralEvidenceRoot",
        "synaptik.cpu.pointwiseMask.performance",
        "synaptik.cpu.pointwiseMask.performanceEvidenceRoot",
        "synaptik.cpu.pointwiseMask.performanceDiagnostic",
        "synaptik.cpu.vectorMse.structuralEvidenceRoot",
        "synaptik.cpu.vectorMse.performance",
        "synaptik.cpu.vectorMse.performanceEvidenceRoot",
        "synaptik.cpu.vectorScalarPower.performance",
        "synaptik.cpu.vectorScalarPower.performanceEvidenceRoot",
        "synaptik.cpu.generalScan.performance",
        "synaptik.cpu.generalScan.pointwisePerformance",
        "synaptik.cpu.generalScan.evidenceRoot"
    ).forEach { name ->
        System.getProperty(name)?.let { systemProperty(name, it) }
    }
    systemProperty("synaptik.cpu.dynamicBatch.performance", System.getProperty("synaptik.cpu.dynamicBatch.performance", "false"))
    System.getenv("SYNAPTIK_CPU_LOSS_PERFORMANCE")?.let {
        environment("SYNAPTIK_CPU_LOSS_PERFORMANCE", it)
    }
}

tasks.named<Javadoc>("javadoc") {
    (options as StandardJavadocDocletOptions).apply {
        memberLevel = JavadocMemberLevel.PACKAGE
        addStringOption("-add-modules", "jdk.incubator.vector")
    }
}

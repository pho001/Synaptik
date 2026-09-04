import org.gradle.api.tasks.compile.JavaCompile
import org.gradle.api.tasks.javadoc.Javadoc
import org.gradle.external.javadoc.StandardJavadocDocletOptions
import org.gradle.external.javadoc.JavadocMemberLevel
import org.gradle.api.tasks.testing.Test

dependencies {
    implementation(project(":modules:model"))
    implementation(project(":modules:config"))
    implementation(project(":modules:planning"))
    implementation(project(":modules:runtime"))
    implementation(project(":modules:prepare"))
    implementation(project(":modules:backend-contract"))
    implementation(project(":modules:trace"))
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
        "synaptik.cpu.loss.c2Evidence"
    ).forEach { name ->
        System.getProperty(name)?.let { systemProperty(name, it) }
    }
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

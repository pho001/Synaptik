import org.gradle.api.tasks.SourceSetContainer
import org.gradle.api.tasks.testing.Test

val trainingApiConsumer = extensions.getByType<SourceSetContainer>().create("trainingApiConsumer")
val nnApiConsumer = extensions.getByType<SourceSetContainer>().create("nnApiConsumer")

dependencies {
    add(trainingApiConsumer.implementationConfigurationName, project(":extensions:training"))
    add(nnApiConsumer.implementationConfigurationName, project(":extensions:nn"))
}

tasks.named<Test>("test") {
    dependsOn(tasks.named(trainingApiConsumer.compileJavaTaskName))
    dependsOn(tasks.named(nnApiConsumer.compileJavaTaskName))
}

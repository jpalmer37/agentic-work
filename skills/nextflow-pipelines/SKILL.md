---
name: nextflow-pipelines
description: Create, review, and improve production-quality bioinformatics pipelines with Nextflow DSL2. Use whenever a task involves Nextflow processes, workflows, channels, nf-core conventions, containers, HPC or cloud executors, resource configuration, testing, provenance, or reproducible pipeline design.
---

# Nextflow Pipelines

Build maintainable DSL2 workflows that are portable across local workstations, HPC schedulers, and cloud executors. Treat reproducibility, validation, and operational failure modes as core design requirements.

## Workflow

1. Clarify inputs, outputs, sample metadata, scale, executor, and acceptance criteria.
2. Separate orchestration from atomic tool processes.
3. Model channels with metadata maps and explicit tuple contracts.
4. Pin every tool through Conda and a container.
5. Configure resources through labels and executor-specific profiles.
6. Add stub runs, small synthetic tests, and execution reports.
7. Record tool versions and workflow parameters with the outputs.

## Project Structure

```text
pipeline/
├── main.nf
├── nextflow.config
├── nextflow_schema.json
├── modules/local/
├── subworkflows/local/
├── conf/
│   ├── base.config
│   └── test.config
├── assets/
├── tests/
└── README.md
```

Keep processes atomic and reusable. Keep workflow composition in `main.nf` or subworkflows, and keep environment-specific settings in profiles rather than hard-coded paths.

## Minimal DSL2 Pattern

```nextflow
nextflow.enable.dsl = 2

params.input = null
params.outdir = 'results'

process FASTQC {
    tag "${meta.id}"
    label 'process_low'

    conda 'bioconda::fastqc=0.12.1'
    container 'quay.io/biocontainers/fastqc:0.12.1--hdfd78af_0'

    input:
    tuple val(meta), path(reads)

    output:
    tuple val(meta), path('*.html'), emit: html
    path 'versions.yml', emit: versions

    script:
    """
    fastqc --threads ${task.cpus} ${reads}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        fastqc: \$(fastqc --version | sed 's/FastQC v//')
    END_VERSIONS
    """

    stub:
    """
    touch ${meta.id}_fastqc.html
    echo '"${task.process}": {fastqc: "0.12.1"}' > versions.yml
    """
}

workflow {
    reads_ch = Channel
        .fromFilePairs(params.input, checkIfExists: true)
        .map { id, reads -> tuple([id: id, single_end: false], reads) }

    FASTQC(reads_ch)
}
```

## Design Rules

- Use DSL2 and named outputs.
- Pass sample identity in a metadata map rather than encoding it in filenames.
- Use tuples to keep metadata and files synchronized.
- Pin exact tool versions in both `conda` and `container` directives.
- Prefer one main tool per process so failures and resource use remain observable.
- Define CPU, memory, and time through labels in configuration.
- Use `publishDir` for final artifacts only; treat `work/` as an execution cache.
- Add `stub:` blocks for fast structural tests.
- Emit `versions.yml` from every tool process.
- Avoid institutional paths, credentials, and account identifiers in public configuration.

## Resource Configuration

```groovy
process {
    withLabel: process_low {
        cpus = 2
        memory = 4.GB
        time = 2.h
    }

    errorStrategy = { task.exitStatus in 137..140 ? 'retry' : 'terminate' }
    maxRetries = 2
}

profiles {
    slurm {
        process.executor = 'slurm'
        singularity.enabled = true
        docker.enabled = false
    }

    test {
        includeConfig 'conf/test.config'
    }
}
```

Cap dynamic retries at an explicit maximum. A retry policy should distinguish resource exhaustion from deterministic tool or input failures.

## Validation Checklist

- `nextflow config` resolves without errors.
- A stub run completes: `nextflow run . -profile test -stub-run`.
- A small synthetic or public test dataset completes end to end.
- Expected outputs, versions, reports, and provenance files are asserted.
- Resume behavior is tested after an intentional interruption.
- At least one target executor profile is exercised outside a developer laptop.
- Documentation states supported versions, inputs, outputs, limitations, and attribution.

## Reference Routing

Read only the references needed for the task:

- [Usage guide](references/usage-guide.md) for common prompts and run commands.
- [Best practices](references/best-practices.md) for process design, schemas, testing, and nf-core conventions.
- [Channel patterns](references/channel-patterns.md) for joins, grouping, branching, and metadata maps.
- [Container management](references/container-management.md) for Conda, BioContainers, registries, and custom images.
- [Version tracking](references/version-tracking.md) for robust `versions.yml` patterns.
- [RNA-seq example](examples/rnaseq.nf) for a compact multi-process workflow.

## Output Expectations

When creating or reviewing a pipeline, return:

1. The proposed file structure and data-flow contract.
2. Complete code for the requested files.
3. A minimal test command and expected outputs.
4. Assumptions, operational limitations, and executor-specific notes.
5. Attribution for adapted modules or shared upstream work.

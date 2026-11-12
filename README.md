# agent-skills

[![Validate skills](https://github.com/jpalmer37/agent-skills/actions/workflows/validate.yml/badge.svg)](https://github.com/jpalmer37/agent-skills/actions/workflows/validate.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-2ea44f)](LICENSE)

A curated collection of reusable instruction packages following the [Agent Skills open standard](https://github.com/anthropics/skills).

## Audience

These skills are for developers and scientific-computing practitioners using coding agents for Python data work, visualization, dashboards, and reproducible Nextflow pipelines. Each directory is independently installable and keeps detailed reference material out of the primary instruction file.

## Skills

| Skill | Description |
|---|---|
| [py-plotting](skills/py-plotting/) | Intuitive and beautiful visualizations with Python Plotly + matplotlib |
| [py-dashboards](skills/py-dashboards/) | Clean and beautiful interactive dashboards using Python Plotly + Dash |
| [py-data](skills/py-data/) | Efficient data analysis using Python pandas and polars |
| [plotly-dash](skills/plotly-dash/) | Connect Plotly figures to production-quality Dash interactions |
| [nextflow-pipelines](skills/nextflow-pipelines/) | Design, validate, and operate reproducible Nextflow DSL2 workflows |

## Learning

[ML Deep Dive](learning/README.md) is a 14-module course for software and data engineers who know the basics of machine learning and want to go deeper: dot products, loss functions, backpropagation, attention, transformers, LLM architectures, evaluation, MLOps, and hosting your own models. It is plain Markdown written to read comfortably on a phone. A separate set of [hands-on labs](learning/labs/README.md) puts it into practice on a local GPU: training a tiny GPT, running and serving open models, retrieval, and LoRA fine-tuning.

## Structure

Each skill lives in its own folder under `skills/` and follows the Agent Skills standard:

```
skills/
└── skill-name/
    ├── SKILL.md          # required — frontmatter metadata + instructions
    ├── references/       # optional — detailed docs loaded only when needed
    ├── scripts/          # optional — deterministic helpers
    └── assets/           # optional — templates and static inputs
```

## Installation

Copy one skill directory into the skills location used by your agent environment. For example:

```bash
cp -R skills/nextflow-pipelines /path/to/your/skills/
```

The exact destination varies by agent runtime. Review a skill before installing it, particularly if it contains scripts.

## Minimal Example

After installing `nextflow-pipelines`, a prompt such as the following should route to it:

> Build a Nextflow DSL2 pipeline that reads paired FASTQ files, runs FastQC in a pinned container, emits `versions.yml`, and includes a SLURM profile plus a stub test.

## Validation

Run the repository validator before opening a pull request:

```bash
python3 scripts/validate_skills.py
```

It verifies frontmatter, directory/name alignment, instruction length, local Markdown references, and the absence of machine-specific absolute paths. GitHub Actions runs the same check for every push and pull request.

## Limitations

- The validator checks packaging and repository hygiene; it does not prove that every generated answer is correct.
- Library and workflow APIs evolve. Confirm version-sensitive guidance against primary documentation before production use.
- Skills provide instructions, not a security boundary. Review generated code and commands before execution.

## Attribution

Designed and maintained by John Palmer. Individual skills may reference external projects and documentation; those projects retain their own names and licenses.

## License

Licensed under the [MIT License](LICENSE).

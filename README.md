# SparseAutoschedulingBenchmark

Sparse autoschedulers are relatively new. This means there’s a golden opportunity to publish the very first sparse autoscheduling benchmark. Similar to https://arxiv.org/abs/2506.02345, this project would involve examining several example applications (from sources like github.com, textbooks, kaggle), and translating them to simple benchmark functions that call standardized high-level sparse operations. The standard form for our benchmark functions in this case will be any vanilla python code which uses Array-API functions https://data-apis.org/array-api/latest/API_specification/. We will also create a database of representative inputs to the benchmarks, and potentially generate inputs automatically. Finally, we will build Array-API compliant frontends for the major sparse autoscheduling frameworks, and compare their performance to determine which is the fastest on real-world inputs. 

## Installation

SparseAutoschedulingBenchmark uses [poetry](https://python-poetry.org/) for packaging. To install for
development, clone the repository and run:
```bash
poetry install --extras test
```
to install the current project and dev dependencies.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, development setup, and best practices.

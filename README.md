# SCOT

Queryable Semantic Compression via Complete One-Hop Transformed Rules.

This package contains one SCOT compression implementation, one query engine,
and mode-selector data collection and training.

```text
src/        SCOT and mode-selector training source
scripts/    Four command-line entry points
datasets/   Rules, queries, selector inputs/model, and a small example
README.md
```

## Install

Use Python 3.9-3.12:

```bash
python -m pip install -r src/requirements.txt
```

The scripts locate the source directory automatically. The database dependency
versions are DataFusion 50.1.0 and PyArrow 15.0.0.

Run the source regression tests with:

```bash
python -m unittest discover -s tests -v
```

Exact dependency versions are listed in `src/requirements.lock.txt`.

## Compress and query

```bash
python scripts/compress.py --facts datasets/example/edge.tsv --rules datasets/example/rules.dlog --out artifacts/chain.json --T 2 --Nb 4 --check
python scripts/query.py --artifact artifacts/chain.json --query "Y :- tc(0,Y)." --mode auto --exact
```

Facts use `predicate<TAB>argument...`. Each rule occupies one line:

```text
tc(X,Y) :- edge(X,Y).
tc(X,Z) :- tc(X,Y), edge(Y,Z).
```

Variables begin with an uppercase letter or `?`. Rules are positive,
function-free, range-restricted Datalog. Numeric and symbolic constants are
supported, including negative and arbitrarily large integers. `1` and `"1"`
are different constants. Database-local integer IDs preserve that distinction.
The version 2 JSON artifact stores original integer/string values, retained
facts and COT rules; version 1 artifacts remain readable.
`--check` verifies closure preservation and single-step recovery.

The query head lists answer variables. A Boolean query has an empty head:
`:- tc(0,7).`. Use `--exact` to return tuples, or omit it to return a count.
Available modes are `p00`, `p01`, `p10`, `p11`, and `auto`. P00 disables both
optimizations, P01 enables UFA, P10 enables QFSE, and P11 enables both. Auto
uses the supplied cost-sensitive policy tree.

For a workload, replace `--query` with `--queries`:

```bash
python scripts/query.py --artifact artifacts/wn18E.json --queries datasets/queries/test/wn18E_test.pkl --mode auto --out results/wn18E.json --memory-gb 256
```

Workloads can be plain text with one CQ per line, JSON, or the supplied
basic-type pickle files. Facts and queries must use matching predicate names
and constant identifiers. Each query uses one warm-up and five measured runs
by default. The timeout is 120 seconds. An anomalous measured run has an
effective cost of 240 seconds. `--warmups`, `--repeats`, and `--timeout` control
these settings. Warm-up measurements are excluded from reported costs.
Warm-up and timed runs reuse one worker session. Failures restart the worker
and trigger another warm-up; failed preparation is explicitly recorded as an
incomplete attempt instead of a successful cold timing. Session identifiers
and worker PIDs are included in the output.

## Train the mode selector

Collect the four modes' costs on a compressed graph:

```bash
python scripts/collect_selector_data.py --artifact artifacts/wn18E.json --queries datasets/queries/train/wn18E_train.pkl --holdout-queries datasets/queries/test/wn18E_test.pkl --dataset-tag wn18E --out artifacts/labels.csv --memory-gb 256
python scripts/train_selector.py --input artifacts/labels.csv --out artifacts/custom.model.json
```

Collection uses one warm-up and five measured runs per query-mode pair.
Timeouts, out-of-memory failures, errors, and worker crashes cost 240 seconds.
The resulting CSV contains the 127 structural features and measured mode costs.
Queries anomalous in all four modes are excluded from fitting.
`--holdout-queries` filters variable-renamed/atom-reordered equivalents of the
specified test queries and duplicate training candidates before collecting new
labels. It can be repeated for multiple holdout files. Exclusions are reported;
input workload files are never changed. Training output records missing feature
columns, source checksums and any supplied label measurement protocol.

Retrain the supplied global model from the bundled 2,747 fitting rows:

```bash
python scripts/train_selector.py --input datasets/selector_training.csv --out artifacts/s2747.model.json
python scripts/query.py --artifact artifacts/chain.json --query "Y :- tc(0,Y)." --model artifacts/s2747.model.json
```

The global model is fitted on WN18-E, FB15K-E, NELL-E, UOBM-U and LUBM-U.
LUBM-L uses this global model without separate training.

## Datasets

| Benchmark | Source and download/generator |
| --- | --- |
| WN18 | [TransE project](https://everest.hds.utc.fr/doku.php?id=en:transe), [WN18 archive](https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz), [authors' repository](https://github.com/glorotxa/SME) |
| FB15K-237 (`fbE`, `fbL`) | [Microsoft release](https://www.microsoft.com/en-us/download/details.aspx?id=52312), [archive](https://download.microsoft.com/download/8/7/0/8700516a-ab3d-4850-b4bb-805c515aece1/FB15K-237.2.zip) |
| NELL-995 | [DeepPath release](https://github.com/xwhan/DeepPath#access-the-dataset), [authors' archive](https://sites.cs.ucsb.edu/~xwhan/datasets/NELL-995.zip) |
| LUBM | [Lehigh project](https://swat.cse.lehigh.edu/projects/lubm/), [UBA 1.7 generator](https://swat.cse.lehigh.edu/projects/lubm/uba1.7.zip) |
| UOBM | [Original IBM benchmark](http://uob.iodt.ibm.com/), [Oxford generator](https://www.cs.ox.ac.uk/isg/tools/UOBMGenerator/), [generator archive](https://www.cs.ox.ac.uk/isg/tools/UOBMGenerator/UOBMGenerator.zip) |

The Oxford UOBM generator produces a different distribution from the original
IBM datasets. These links identify upstream data and generators. The ten rule
settings are under `datasets/rules/`. Each final test workload contains 100
queries. Training query files cover the five global-selector fitting settings.
Large graphs and compression artifacts are stored outside this package.

gMark: https://github.com/gbagan/gmark

## License

Project code is provided under the [MIT License](LICENSE). Third-party datasets
and dependencies retain their own terms. Please cite the SCOT paper in research.

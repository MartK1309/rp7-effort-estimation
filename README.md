# Story Point Estimation Replication Package

This repository contains the replication package for the paper "Investigating the Effectiveness of Similarity-Based Retrieval for Story Point Estimation". The study compares different methods for estimating story points in agile software development projects, using the [TAWOS dataset](https://github.com/SOLAR-group/TAWOS).

## Project Overview

The project implements and evaluates various story point estimation methods across multiple software projects. It includes:

- Implementation of semantic similarity-based estimation approaches
- Comparison with baseline and state-of-the-art methods
- Evaluation using standard metrics (MAE, MdAE, SA)
- Analysis across multiple open-source projects

## Dataset

The dataset (TAWOS) includes user stories from 23 different projects:
- ALOY, APSTUD, CLI, CLOV, COMPASS, CONFCLOUD, DAEMON, DM, DNN, DURACLOUD
- EVG, FAB, MDL, MESOS, MULE, NEXUS, SERVER, STL, TIDOC, TIMOB, TISTUD, XD

Each project's data is split into:
- Training set (`project-train.csv`)
- Validation set (`project-valid.csv`)
- Test set (`project-test.csv`)

With corresponding feature files (`project-*_features.csv`).

## Repository Structure

```
.
├── app.py                 # Main application file
├── data/
│   ├── TAWOS/            # Dataset files
│   └── comparison/       # Comparison implementation and results
├── helpers/              # Utility functions
│   ├── descriptive_stats.py
│   ├── evaluation.py
│   ├── text_preprocessing.py
│   └── weaviate.py
└── output/              # Output files and results
```

## Methods Implemented

The package implements and compares several story point estimation methods:
- LHC-SE (Language-based Historical Consistency)
- LHC_TC-SE (LHC with Type Consistency)
- Deep-SE
- TF-IDF-SE
- Baseline methods (Mean, Median)

## Metrics

The following metrics are used for evaluation:
- MAE (Mean Absolute Error)
- MdAE (Median Absolute Error)
- SA (Standardized Accuracy)

## Requirements

- Python 3.x
- Docker
## Setup

1. Clone the repository
2. Duplicate .env.example and rename to `.env`
3. For running SBERT-SB-SE only, `USE_LLM_EMBEDDINGS` in the .env file can be kept to false. This way, the OPENAI_API_KEY does not have to be present. 
3. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Start the Weaviate instance:
```bash
docker-compose up -d
```

## Running the Experiments

The main experimental pipeline can be executed through `app.py`:

```bash
python app.py
```

This will:
1. Load the dataset
2. Generate & upsert embeddings from training seet
4. Perform story point estimation on testing set
5. Evaluate results and generate comparison tables

## Results

Results are generated in the `output/` directory:
- `comparison_table.tex`: Detailed comparison across all methods

## License

[Add appropriate license information]
MIT

## Citation

[Add citation information for the research paper]

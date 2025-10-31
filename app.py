import weaviate
from helpers.descriptive_stats import show_project_metrics
from helpers.weaviate import createCollection, upsertProject
import pingouin as pg
from helpers.evaluation import (
    plot_vector_variances,
    evaluate_project,
    evaluate_vectors,
    generate_overview_table,
    create_comparison_table,
)
from weaviate.classes.init import AdditionalConfig, Timeout
import matplotlib

matplotlib.use("Agg")
import numpy as np
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

USE_LLM_EMBEDDINGS = os.getenv("USE_LLM_EMBEDDINGS", "false").lower() == "true"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
client = weaviate.connect_to_local(
    additional_config=AdditionalConfig(timeout=Timeout(init=30, query=120, insert=600))
)

project_list = [
    "ALOY",
    "APSTUD",
    "CLI",
    "CLOV",
    "COMPASS",
    "CONFCLOUD",
    "DAEMON",
    "DM",
    "DNN",
    "DURACLOUD",
    "EVG",
    "FAB",
    "MDL",
    "MESOS",
    "MULE",
    "NEXUS",
    "SERVER",
    "STL",
    "TIDOC",
    "TIMOB",
    "TISTUD",
    "XD",
]
project_keys = ["DM"]

project_keys = project_list
# Data
try:
    # Uncomment to reset existing weaviate collection
    # client.collections.delete("UserStoryCollection")
    if client.collections.exists("UserStoryCollection"):
        collection = client.collections.use("UserStoryCollection")
    else:
        collection = createCollection(client)
    # Uncomment to show project metrics
    # show_project_metrics(project_list)

    for project_key in project_keys:
        collection = upsertProject(collection, project_key)

    results_data_SBERT = []
    vector_variances_SBERT = {}
    train_coverages_SBERT = {}

    results_data_llm = []
    vector_variances_llm = {}
    train_coverages_llm = {}

    results_lock = threading.Lock()

    def evaluate_both(project_key):
        try:
            MAEpi_sbert, MdAE_sbert, SA_sbert, coverage_sbert = evaluate_project(
                project_key, collection, similarity_threshold=SIMILARITY_THRESHOLD, vectorizer="miniLM_vector"
            )
            sbert_result = [project_key, MAEpi_sbert, MdAE_sbert, SA_sbert, coverage_sbert]

            if USE_LLM_EMBEDDINGS:
                MAEpi_llm, MdAE_llm, SA_llm, coverage_llm = evaluate_project(
                    project_key, collection, similarity_threshold=SIMILARITY_THRESHOLD, vectorizer="openai_vector"
                )
                llm_result = [project_key, MAEpi_llm, MdAE_llm, SA_llm, coverage_llm]
                sbert_mean_variance, sbert_coverage, llm_mean_variance, llm_coverage = (
                    evaluate_vectors(collection, project_key, SIMILARITY_THRESHOLD)
                )

                # Return a consistent 6-tuple: (project_key, sbert_variance, train_coverage_sbert, llm_variance, train_coverage_llm, exc)
                return (
                    project_key,
                    sbert_mean_variance,
                    sbert_coverage,
                    llm_mean_variance,
                    llm_coverage,
                    sbert_result,
                    llm_result,
                    None
                )
            # If not using LLM embeddings, return same 6-tuple shape with LLM fields and exc set to None
            return (project_key, None, None, None, None, None)

        except Exception as e:
            return (project_key, None, None, None, None, e)

    # Submit all projects concurrently and collect results
    futures = []
    with ThreadPoolExecutor(max_workers=min(8, len(project_keys))) as executor:
        for project_key in project_keys:
            futures.append(executor.submit(evaluate_both, project_key))

        for fut in as_completed(futures):
            (
                project_key,
                sbert_mean_variance,
                sbert_coverage,
                llm_mean_variance,
                llm_coverage,
                sbert_result,
                llm_result,
                exc,
            ) = fut.result()
            if exc is not None:
                print(f"Error evaluating project {project_key}: {exc}")
                continue
            with results_lock:
                results_data_SBERT.append(sbert_result)
                vector_variances_SBERT[project_key] = sbert_mean_variance
                train_coverages_SBERT[project_key] = sbert_coverage
                if USE_LLM_EMBEDDINGS:
                    vector_variances_llm[project_key] = llm_mean_variance
                    train_coverages_llm[project_key] = llm_coverage
                    if llm_result is not None:
                        results_data_llm.append(llm_result)
                    
    projects = list(vector_variances_SBERT.keys())
    values_sbert = [vector_variances_SBERT[p] for p in projects]

    # Generate LateX tables
    # generate_overview_table(results_data_SBERT)
    # create_comparison_table(results_data_SBERT, "SBERT-SB-SE")
    if USE_LLM_EMBEDDINGS:
        plot_vector_variances(vector_variances_SBERT, vector_variances_llm)
        values_llm = [vector_variances_llm[p] for p in projects]
        mean_sbert, std_sbert = np.mean(values_sbert), np.std(values_sbert)
        mean_llm, std_llm = np.mean(values_llm), np.std(values_llm)

        wilcoxon_stats = pg.wilcoxon(values_sbert, values_llm)
        print(f"Wilcoxon test for Similarity threshold {SIMILARITY_THRESHOLD}:")
        print(wilcoxon_stats)
        print(f"SBERT mean variance for Similarity threshold {SIMILARITY_THRESHOLD}: {mean_sbert:.3f} ± {std_sbert:.3f}")
        print(f"LLM mean variance for Similarity threshold {SIMILARITY_THRESHOLD}: {mean_llm:.3f} ± {std_llm:.3f}")


finally:
    client.close()

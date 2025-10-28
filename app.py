import weaviate
from helpers.descriptive_stats import show_project_metrics
from helpers.weaviate import createCollection, upsertProject, estimateStorypoint
from helpers.evaluation import plot_vector_variances, evaluate_project, evaluate_vectors, generate_overview_table, create_comparison_table
from weaviate.classes.init import AdditionalConfig, Timeout
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

USE_LLM_EMBEDDINGS = os.getenv('USE_LLM_EMBEDDINGS', 'false').lower() == 'true'

client = weaviate.connect_to_local(
    additional_config=AdditionalConfig(
        timeout=Timeout(
            init=30, query=120, insert=600
        ) 
    )
)
project_keys = ["DM"]

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
    "XD"    
]
project_keys = project_list
# Data
try:
    # Uncomment to reset existing weaviate collection
    client.collections.delete("UserStoryCollection")
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

    results_data_llm = []
    vector_variances_llm = {}

    results_lock = threading.Lock()

    def evaluate_both(project_key):
        try:
            MAEpi_sbert, MdAE_sbert, SA_sbert, coverage_sbert = evaluate_project(
                project_key, collection, certainty=0.8, vectorizer="miniLM_vector"
            )
            sbert_result = [project_key, MAEpi_sbert, MdAE_sbert, SA_sbert, coverage_sbert]
            sbert_variance, _ = evaluate_vectors(collection, project_key, vector="miniLM_vector")

            if USE_LLM_EMBEDDINGS:
                MAEpi_llm, MdAE_llm, SA_llm, coverage_llm = evaluate_project(
                    project_key, collection, certainty=0.8, vectorizer="openai_vector"
                )
                llm_result = [project_key, MAEpi_llm, MdAE_llm, SA_llm, coverage_llm]
                llm_variance, _ = evaluate_vectors(collection, project_key, vector="openai_vector")
                return (project_key, sbert_result, sbert_variance, llm_result, llm_variance, None)

            return (project_key, sbert_result, sbert_variance, None, None, None)
        except Exception as e:
            # Return the exception so the caller can log/handle it without losing other results
            return (project_key, None, None, None, None, e)

    # Submit all projects concurrently and collect results
    futures = []
    with ThreadPoolExecutor(max_workers=min(8, len(project_keys))) as executor:
        for project_key in project_keys:
            futures.append(executor.submit(evaluate_both, project_key))

        for fut in as_completed(futures):
            project_key, sbert_result, sbert_variance, llm_result, llm_variance, exc = fut.result()
            if exc is not None:
                print(f"Error evaluating project {project_key}: {exc}")
                continue
            with results_lock:
                results_data_SBERT.append(sbert_result)
                vector_variances_SBERT[project_key] = sbert_variance
                if USE_LLM_EMBEDDINGS and llm_result is not None:
                    results_data_llm.append(llm_result)
                    vector_variances_llm[project_key] = llm_variance

    projects = list(vector_variances_SBERT.keys())
    values_sbert = [vector_variances_SBERT[p] for p in projects]

    # Generate LateX tables
    generate_overview_table(results_data_SBERT)
    create_comparison_table(results_data_SBERT, "SBERT-SB-SE")
    
    if USE_LLM_EMBEDDINGS:
        plot_vector_variances(vector_variances_SBERT, vector_variances_llm)
        values_llm = [vector_variances_llm[p] for p in projects]
        wilcoxon_stat, wilcoxon_p = wilcoxon(values_sbert, values_llm)
        mean_llm, std_llm = np.mean(values_llm), np.std(values_llm)
        print(f"LLM mean variance: {mean_llm:.3f} ± {std_llm:.3f}")
        print(f"Wilcoxon W={wilcoxon_stat:.3f}, p={wilcoxon_p:.4f}")


finally:
    client.close()

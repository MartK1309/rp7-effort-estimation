import weaviate
from helpers.weaviate import createCollection, upsertProject, estimateStorypoint
from helpers.evaluation import evaluate_project, evaluate_vectors, generate_overview_table, create_comparison_table
from weaviate.classes.init import AdditionalConfig, Timeout
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    # client.collections.delete("UserStoryCollection")
    if client.collections.exists("UserStoryCollection"):
        collection = client.collections.use("UserStoryCollection")
    else:
        collection = createCollection(client)
    # show_project_metrics(project_list)
    # raise Exception("Stop execution after showing project metrics")
    # Only run once when upserting:
    # collection = upsertProject(collection, project_key)

    # for project_key in project_keys:
    #     collection = upsertProject(collection, project_key)

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

            MAEpi_llm, MdAE_llm, SA_llm, coverage_llm = evaluate_project(
                project_key, collection, certainty=0.8, vectorizer="openai_vector"
            )
            llm_result = [project_key, MAEpi_llm, MdAE_llm, SA_llm, coverage_llm]
            llm_variance, _ = evaluate_vectors(collection, project_key, vector="openai_vector")

            return (project_key, sbert_result, sbert_variance, llm_result, llm_variance, None)
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
                # Log the error and continue; keep data structures consistent
                print(f"Error evaluating project {project_key}: {exc}")
                continue
            # Use a lock to safely update shared structures
            with results_lock:
                results_data_SBERT.append(sbert_result)
                vector_variances_SBERT[project_key] = sbert_variance
                results_data_llm.append(llm_result)
                vector_variances_llm[project_key] = llm_variance
    projects = list(vector_variances_llm.keys())
    values_llm = [vector_variances_llm[p] for p in projects]
    values_sbert = [vector_variances_SBERT[p] for p in projects]
    wilcoxon_stat, wilcoxon_p = wilcoxon(values_sbert, values_llm)

    # Descriptive stats
    mean_llm, std_llm = np.mean(values_llm), np.std(values_llm)
    mean_sbert, std_sbert = np.mean(values_sbert), np.std(values_sbert)

    print(f"SBERT mean variance: {mean_sbert:.3f} ± {std_sbert:.3f}")
    print(f"LLM mean variance: {mean_llm:.3f} ± {std_llm:.3f}")
    print(f"Wilcoxon W={wilcoxon_stat:.3f}, p={wilcoxon_p:.4f}")

    x = np.arange(len(projects))  # positions for groups
    width = 0.4  # width of each bar

    # Create the bars
    plt.bar(x - width/2, values_sbert, width, label='SBERT', color='lightcoral')
    plt.bar(x + width/2, values_llm, width, label='LLM', color='cornflowerblue')

    # Labels and formatting
    plt.ylabel("Average Local Story Point Std. Dev.")
    plt.xlabel("Project")
    plt.title("Average Local Story Point Variance per Project")
    plt.xticks(x, projects, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    generate_overview_table(results_data_SBERT)
    create_comparison_table(results_data_SBERT, "SBERT-SB-SE")

    # Save and show
    plt.savefig("./output/vector_variance_per_project.png", dpi=300)
    plt.show()
    # plt.figure(figsize=(10, 6))
    # projects = list(vector_variances_llm.keys())
    # values = list(vector_variances_llm.values())

    # plt.bar(projects, values, color='cornflowerblue')
    # # plt.axhline(2.0, color='red', linestyle='--', label='High Variance Threshold (2.0)')
    # plt.ylabel("Average Local Story Point Std. Dev.")
    # plt.xlabel("Project").
    # plt.title("Average Local Story Point Variance per Project")
    # plt.xticks(rotation=45, ha='right')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(f"./output/vector_variance_per_project.png")

    # create_comparison_table(results_data_llm, "OPENAI-SB-SE")

finally:
    client.close()

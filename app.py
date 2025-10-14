import weaviate
from helpers.weaviate import createCollection, upsertProject, estimateStorypoint
from helpers.evaluation import evaluate_project, evaluate_vectors, generate_overview_table, create_comparison_table
from weaviate.classes.init import AdditionalConfig, Timeout
import matplotlib.pyplot as plt
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

    for project_key in project_keys:
        collection = upsertProject(collection, project_key)
    results_data = []
    vector_variances = {}
    for project_key in project_keys:
        MAEpi, MdAE, SA, coverage = evaluate_project(project_key, collection, certainty=0.7)
        # Append the results for the current project to the list
        results_data.append([project_key, MAEpi, MdAE, SA, coverage])
        vector_variances[project_key], train_coverage = evaluate_vectors(collection, project_key)

    plt.figure(figsize=(10, 6))
    projects = list(vector_variances.keys())
    values = list(vector_variances.values())

    generate_overview_table(results_data)
    create_comparison_table(results_data, "SBERT-SB-SE")

    plt.bar(projects, values, color='cornflowerblue')
    # plt.axhline(2.0, color='red', linestyle='--', label='High Variance Threshold (2.0)')
    plt.ylabel("Average Local Story Point Std. Dev.")
    plt.xlabel("Project")
    plt.title("Average Local Story Point Variance per Project")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"./output/vector_variance_per_project.png")
finally:
    client.close()

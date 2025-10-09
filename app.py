import weaviate
from helpers.weaviate import createCollection, upsertProject, estimateStorypoint
from helpers.evaluation import evaluate_project, visualize_vectors
from weaviate.classes.init import AdditionalConfig, Timeout

client = weaviate.connect_to_local(
    additional_config=AdditionalConfig(
        timeout=Timeout(
            init=30, query=120, insert=600
        ) 
    )
)
project_key = "EVG"
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

# Data
try:
    client.collections.delete("UserStoryCollection")
    if client.collections.exists("UserStoryCollection"):
        collection = client.collections.use("UserStoryCollection")
    else:
        collection = createCollection(client)
    # TODO: Make sure to only return collection if already present
    # Only run once when upserting:
    # collection = upsertProject(collection, project_key)

    # items = client.collections.use("UserStoryCollection").query.fetch_objects(limit=10)
    # item = collection.query.near_text("Custom framework and widget", limit=5)
    results_data = []
    for project_key in project_list[:5]:
        collection = upsertProject(collection, project_key)
    for project_key in project_list[:5]:
        MAEpi, MdAE, SA = evaluate_project(project_key, collection)
        # Append the results for the current project to the list
        results_data.append([project_key, MAEpi, MdAE, SA])
        visualize_vectors(collection, project_key)

    latex_table = """
    \\begin{table}[h!]
    \\centering
    \\begin{tabular}{|l|c|c|c|}
    \\hline
    \\textbf{Project Key} & \\textbf{MAE} & \\textbf{MdAE} & \\textbf{SA} \\\\
    \\hline
    """

    # Add a row for each project in your results
    for row in results_data:
        # We use an f-string to format the numbers to 3 decimal places
        # The .replace('_', '\\_') is important to escape underscores for LaTeX
        project_key_escaped = str(row[0]).replace('_', '\\_')
        latex_table += f"{project_key_escaped} & {row[1]:.3f} & {row[2]:.3f} & {row[3]:.3f} \\\\\n"

    # Add the closing lines for the table
    latex_table += """\\hline
    \\end{tabular}
    \\caption{Evaluation metrics for each project.}
    \\label{tab:project_metrics}
    \\end{table}
    """

    # Print the final LaTeX code
    print(latex_table)
    with open(f"./output/all_projects_results.tex", "w") as f:
        f.write(latex_table.strip())
finally:
    client.close()

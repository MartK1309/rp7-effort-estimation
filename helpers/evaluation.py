import numpy as np
import pandas as pd
import umap
from weaviate.collections import Collection
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import random
from helpers.weaviate import estimateStorypoint
from typing import Literal
from sklearn.neighbors import NearestNeighbors
from data.comparison.comparison_data import data as existing_method_results
import os
from collections import defaultdict
from scipy.stats import wilcoxon


def random_guessing_mae(y_true_list, n_runs=1000, random_state=None):
    """
    Compute MAE_p0 for random guessing baseline as described in Tawosi et al. (2022)
    Each run randomly assigns each target case a story point from another
    (different) case in the dataset.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true_list)
    maes = []

    for _ in range(n_runs):
        perm = np.arange(n)
        # shuffle until no element matches its original position
        while True:
            rng.shuffle(perm)
            if not np.any(perm == np.arange(n)):
                break
        y_pred = y_true_list[perm]
        maes.append(np.mean(np.abs(y_true_list - y_pred)))

    return np.mean(maes)

def generate_overview_table(results_data):
    latex_table = """
    \\begin{table}[h!]
    \\centering
    \\begin{tabular}{|l|c|c|c|c|}
    \\hline
    \\textbf{Project Key} & \\textbf{MAE} & \\textbf{MdAE} & \\textbf{SA} & \\textbf{Coverage} \\\
    \\hline
    """

    # Add a row for each project in your results
    for row in results_data:
        project_key_escaped = str(row[0]).replace("_", "\\_")
        coverage_raw = row[4] if len(row) > 4 else 0.0
        try:
            coverage_val = float(coverage_raw)
        except Exception:
            coverage_val = 0.0

        # Accept either fraction (0-1) or percentage (0-100)
        if coverage_val <= 1:
            coverage_pct = coverage_val * 100
        else:
            coverage_pct = coverage_val

        # Format coverage with one decimal and escape percent symbol for LaTeX
        coverage_str = f"{coverage_pct:.1f}\\%"

        latex_table += (
            f"{project_key_escaped} & {row[1]:.3f} & {row[2]:.3f} & {row[3]:.3f} & {coverage_str} \\\\n+"
        )

    # Add the closing lines for the table
    latex_table += """\\hline
  \\end{tabular}
  \\caption{Evaluation metrics for each project.}
  \\label{tab:project_metrics}
  \\end{table}
  """
    with open(f"./output/all_projects_results.tex", "w") as f:
        f.write(latex_table.strip())


def evaluate_project(
    project_key: str,
    collection: Collection,
    certainty=0.8,
    vectorizer: Literal["openai_vector", "miniLM_vector"] = "miniLM_vector",
):
    errors = []
    df_train = pd.read_csv(f"./data/TAWOS/{project_key}-train.csv")
    project_median = int(df_train["storypoint"].median())

    df_testset = pd.read_csv(f"./data/TAWOS/{project_key}-test.csv")
    no_similar_stories = 0
    # Loop through testset
    for index, row in df_testset.iterrows():
        estimate_sp = estimateStorypoint(
            collection,
            title=row["title"],
            description=row["description_text"],
            type=row["type"],
            projectName=project_key,
            certainty=certainty,
            vectorizer=vectorizer,
            components=row["components"],
        )
        true_sp = row["storypoint"]
        if estimate_sp is None:
            # Add median of project
            no_similar_stories += 1
            errors.append(abs(true_sp - project_median))
        else:
            # Calculate absolute error
            errors.append(abs(true_sp - estimate_sp))
    print(
        f"Out of {len(df_testset)} there were {no_similar_stories} with no similar stories, so for those the project median of {project_median} was used"
    )
    MAEpi = np.mean(errors)
    MdAE = np.median(errors)

    # Calculate Standard Accuracy
    y_true = df_testset["storypoint"].to_numpy()
    # Keep random seed (random_state) for reproducibility
    MAEp0 = random_guessing_mae(y_true, n_runs=1000, random_state=42)
    SA = (1 - MAEpi / MAEp0) * 100
    coverage = ((len(df_testset) - no_similar_stories) / len(df_testset)) * 100
    print(
        f"""
      Evaluation Report for Project {project_key}
      -------------------------------------------
      Mean Absolute Error (MAE): {MAEpi:.2f}
      Median Absolute Error (MdAE): {MdAE:.2f}
      Standard Accuracy (SA): {SA:.2f}%
      Coverage: {coverage:.2f}%
      """
    )
    return MAEpi, MdAE, SA, coverage


def calculate_local_variance(vectors: np.ndarray, storypoints: np.ndarray, n_neighbors=5, similarity_threshold=0.7):
    # Fit nearest neighbors model
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine").fit(vectors)
    distances, indices = nbrs.kneighbors(vectors)

    # Convert cosine distance to cosine similarity
    similarities = 1 - distances

    variances = []
    for i, idxs in enumerate(indices):
        sims = similarities[i][1:]  # exclude self
        neighbors = idxs[1:]  # exclude self

        # Filter neighbors by similarity threshold
        mask = sims >= similarity_threshold
        if not np.any(mask):
            continue  # skip stories with no sufficiently similar neighbors

        neighbor_points = storypoints[neighbors][mask]
        var = np.std(neighbor_points)
        variances.append(var)

    if len(variances) == 0:
        return np.nan, 0 

    return np.mean(variances), len(variances) / len(vectors)

def evaluate_vectors(collection: Collection, project_key: str, vector: Literal["openai_vector", "miniLM_vector"] = "miniLM_vector"):
    vectors, storypoints = [], []
    collection = collection.with_tenant(project_key)
    for obj in collection.iterator(
        include_vector=True, return_properties=["storypoint"]
    ):
        vectors.append(obj.vector[vector])
        storypoints.append(obj.properties["storypoint"])
    vectors_np, storypoints_np = np.array(vectors), np.array(storypoints)
    mean_variance, train_coverage = calculate_local_variance(vectors_np, storypoints_np, n_neighbors=5)
    umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=3, random_state=42)
    vectors_3d = umap_model.fit_transform(vectors_np)
    vectors_3d = np.array(vectors_3d)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        vectors_3d[:, 0],
        vectors_3d[:, 1],
        vectors_3d[:, 2], # type: ignore
        c=storypoints,
        cmap="viridis",
        s=50,
    )
    fig.colorbar(scatter, label="Storypoint", ax=ax)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    plt.title("UserStory Embeddings Colored by Storypoint")
    plt.savefig(f"./output/{project_key}_umap_3d.png")
    plt.close()
    return mean_variance, train_coverage


def save_as_latex(data, output_path):
    """
    Generates a LaTeX table from the data and saves it to a .tex file.

    Args:
      data (list): The list of result dictionaries to include in the table.
      output_path (str): The path to save the output .tex file,
                         e.g., './output/comparison_table.tex'.
    """
    # 1. Group the data by project for structured table generation
    grouped_data = defaultdict(list)
    for row in data:
        grouped_data[row["Project"]].append(row)

    # Get a sorted list of project names for consistent table order
    sorted_projects = sorted(grouped_data.keys())

    # 2. Begin building the LaTeX string with a document preamble
    # This creates a complete, compilable .tex file.
    latex_parts = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",  # For professional-looking rules (\toprule, etc.)
        r"\usepackage[margin=1in]{geometry}",  # To set reasonable margins
        r"\begin{document}",
        r"",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Comparison of Prediction Methods Across All Projects}",
        r"  \label{tab:comparison}",
        r"  \begin{tabular}{l c c c}",
        r"    \toprule",
        r"    \textbf{Method} & \textbf{MAE} & \textbf{MdAE} & \textbf{SA} \\",
        r"    \midrule",
    ]

    # 3. Create the table body by iterating through each project
    for i, project in enumerate(sorted_projects):
        # Add a clear header row for the project
        latex_parts.append(
            rf"    \multicolumn{{4}}{{c}}{{\textit{{{project}}}}} \\ \midrule"
        )

        # Add a row for each method within that project
        for row in grouped_data[project]:
            # Sanitize method name for LaTeX (e.g., escape underscores)
            method = row["Method"].replace("_", r"\_")
            # Format numbers to two decimal places
            mae = f"{row['MAE']:.2f}"
            mdae = f"{row['MdAE']:.2f}"
            sa = f"{row['SA']:.2f}"
            latex_parts.append(f"    {method} & {mae} & {mdae} & {sa} \\\\")

        # Add a line separating project groups
        if i < len(sorted_projects) - 1:
            latex_parts.append(r"    \midrule")

    # 4. Add the LaTeX postamble to close the environments
    latex_parts.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            r"",
            r"\end{document}",
        ]
    )

    # 5. Write the complete string to the specified file
    try:
        # Ensure the output directory exists before trying to save the file
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Create directory if path is not in the root
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("\n".join(latex_parts))

        print(f"✅ LaTeX table successfully saved to: {output_path}")

    except Exception as e:
        print(f"❌ An error occurred while saving the file: {e}")


def create_comparison_table(result_list, method_name):
    formatted_new_results = []
    for result_item in result_list:
        project_key = result_item[0]
        mae = result_item[1]
        mdae = result_item[2]
        sa = result_item[3]
        
        result_dict = {
            'Project': project_key,
            'Method': method_name,
            'MAE': mae,
            'MdAE': mdae,
            'SA': sa
        }
        formatted_new_results.append(result_dict)
    updated_data = existing_method_results.copy()
    updated_data.extend(formatted_new_results)
    save_as_latex(updated_data, "./output/comparison_table.tex")
    return updated_data

def plot_vector_variances(vector_variances_SBERT, vector_variances_llm):
    projects = list(vector_variances_SBERT.keys())
    values_sbert = [vector_variances_SBERT[p] for p in projects]
    values_llm = [vector_variances_llm[p] for p in projects]
    x = np.arange(len(projects))  
    width = 0.4 
    plt.bar(x - width/2, values_sbert, width, label='SBERT', color='lightcoral')
    plt.bar(x + width/2, values_llm, width, label='LLM', color='cornflowerblue')
    plt.ylabel("Average Local Story Point Std. Dev.")
    plt.xlabel("Project")
    plt.title("Average Local Story Point Variance per Project")
    plt.xticks(x, projects, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig("./output/vector_variance_per_project.png", dpi=300)
    plt.figure(figsize=(10, 6))
    plt.close()

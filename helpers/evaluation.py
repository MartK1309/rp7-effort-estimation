import numpy as np
import pandas as pd
import umap
from weaviate.collections import Collection
import matplotlib.pyplot as plt

from helpers.weaviate import estimateStorypoint


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


# Generate a overleaf table with the results
def generate_overleaf_table(project_key: str, MAE, MdAE, SA):
    table = rf"""
\begin{{table}}[h]
\centering
\begin{{tabular}}{{|c|c|c|c|}}
\hline
Project & MAE & MdAE & SA \\
\hline
{project_key} & {MAE:.2f} & {MdAE:.2f} & {SA:.2f}\% \\
\hline
\end{{tabular}}
\caption{{Evaluation results for project {project_key}}}
\label{{tab:{project_key}_results}}
\end{{table}}
    """
    print(table)
    # It's good practice to .strip() the string to remove leading/trailing whitespace
    with open(f"./output/{project_key}_results.tex", "w") as f:
        f.write(table.strip())
    return table


def evaluate_project(project_key: str, collection: Collection):
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
            certainty=0.7,
            vectorizer="miniLM_vector",
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
    print(
        f"""
      Evaluation Report for Project {project_key}
      -------------------------------------------
      Mean Absolute Error (MAE): {MAEpi:.2f}
      Median Absolute Error (MdAE): {MdAE:.2f}
      Standard Accuracy (SA): {SA:.2f}%
      """
    )
    generate_overleaf_table(project_key, MAEpi, MdAE, SA)
    return MAEpi, MdAE, SA


def visualize_vectors(collection: Collection, project_key: str):
    vectors = []
    storypoints = []
    collection = collection.with_tenant(project_key)
    umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=3, random_state=42)
    for obj in collection.iterator(
        include_vector=True, return_properties=["storypoint"]
    ):
        vectors.append(obj.vector["miniLM_vector"])
        storypoints.append(obj.properties["storypoint"])
    vectors_np = np.array(vectors)
    vectors_3d = umap_model.fit_transform(vectors_np)
    vectors_3d = np.array(vectors_3d)  # Ensure it's a NumPy array for slicing

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
    plt.show()

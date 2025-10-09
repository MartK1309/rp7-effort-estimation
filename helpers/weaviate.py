import pandas as pd
import numpy as np

from weaviate import WeaviateClient
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.collections import Collection
from weaviate.util import generate_uuid5
from helpers.text_preprocessing import preprocess

def createCollection(client: WeaviateClient):
    print("Creating collection 'UserStoryCollection'")
    client.collections.create(
        "UserStoryCollection",
        vector_config=[
            Configure.Vectors.text2vec_transformers(
                name="miniLM_vector",
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                ),
            ),
            # Uncomment for LLM vectorization
            # Configure.Vectors.text2vec_openai(
            #     name="openai_vector",
            #     model="text-embedding-3-small",
            #     vector_index_config=Configure.VectorIndex.hnsw(
            #         distance_metric=VectorDistances.COSINE
            #     ),
            # ),
        ],
        multi_tenancy_config=Configure.multi_tenancy(
            enabled=True, auto_tenant_creation=True
        ),
        properties=[
            Property(
                name="title",
                data_type=DataType.TEXT,
                description="The title of the user story",
            ),
            Property(
                name="description",
                data_type=DataType.TEXT,
                description="The description of the user story",
            ),
            Property(
                name="type",
                data_type=DataType.TEXT,
                description="The type of the user story",
            ),
            Property(
                name="storypoint",
                data_type=DataType.INT,
                description="The story points assigned to the user story",
                skip_vectorization=True,
            ),
        ],
    )
    return client.collections.use("UserStoryCollection")


def upsertProject(collection: Collection, projectName: str):
    print(f"Upserting project {projectName}")
    collection = collection.with_tenant(projectName)
    try:
        df = pd.read_csv(f"./data/TAWOS/{projectName}-train.csv")
        with collection.batch.fixed_size(batch_size=30) as batch:
            for _, row in df.iterrows():
                # Preprocess title and description according to the same steps as in LHC-SE
                preprocessed_title = preprocess(row["title"])
                preprocessed_description = preprocess(row["description_text"])
                obj = {
                    "title": preprocessed_title,
                    "description": preprocessed_description,
                    "storypoint": int(row["storypoint"]),
                    "type": row["type"],
                }
                batch.add_object(uuid=generate_uuid5(row["issuekey"]), properties=obj)
            if batch.number_errors > 10:
                print("Batch import stopped due to excessive errors.")
        failed_objects = collection.batch.failed_objects
        if failed_objects:
            print(f"Number of failed imports: {len(failed_objects)}")
            print(f"First failed object: {failed_objects[0]}")
    # Catch error
    except FileNotFoundError:
        print(f"Project with key {projectName} was not found. Skipping...")
    except Exception as e:
        print(f"Error when upserting project {projectName}: {e}")
    return collection
    # Get first 10 rows to verify
    # items = client.collections.use("UserStoryCollection").query.fetch_objects(limit=10)
    # item = collection.query.near_text("Custom framework and widget", limit=5)
    # print(item.objects)
highest_certainty = 0
# TODO: Move to different module
def estimateStorypoint(
    collection: Collection,
    title: str,
    description: str,
    projectName: str,
    type: str,
    certainty=0.8,
    vectorizer="miniLM_vector",
):
    collection = collection.with_tenant(projectName)
    # For single property search
    # result = collection.query.near_text(text, limit=5, certainty=.8)
    # Multiple properties search
    # Weaviate automatically sorts by alphabet, to description comes before title
    # result = collection.query.near_text(
    #     query=description + " " + title + " " + type, target_vector="miniLM_vector", certainty=certainty, limit=10,return_metadata=["certainty"]
    # )
    result = collection.query.near_text(
        query=description + " " + title + " " + type,
        target_vector=vectorizer,
        certainty=certainty,
        limit=10,
        return_metadata=["certainty"],
    )
    print(f"found {len(result.objects)} similar stories with confidence > {certainty}")
    weights = []

    if result.objects:
        storypoints = []
        weights = []

        for obj in result.objects:
            storypoint_str = str(obj.properties["storypoint"]).strip()
            similarity = obj.metadata.certainty

            if (
                storypoint_str != ""
                and storypoint_str.lower() != "nan"
                and similarity is not None
            ):
                storypoint = int(storypoint_str)
                storypoints.append(storypoint)
                weights.append(similarity)
                print(f"Storypoint raw: '{storypoint_str}', similarity: {similarity}")
                global highest_certainty
                if similarity > highest_certainty:
                    highest_certainty = similarity
                    print(f"New highest certainty: {highest_certainty}")

        if storypoints and weights and sum(weights) > 0 and similarity is not None:
            # sort storypoints and corresponding weights
            sorted_indices = np.argsort(storypoints)
            sorted_storypoints = np.array(storypoints)[sorted_indices]
            sorted_weights = np.array(weights)[sorted_indices]

            # compute cumulative sum of weights and normalize to 1
            cumulative_weights = np.cumsum(sorted_weights)
            cutoff = cumulative_weights[-1] / 2.0

            # find the storypoint where cumulative weight crosses 50%
            weighted_median = sorted_storypoints[np.searchsorted(cumulative_weights, cutoff)]

            return int(weighted_median)
        else:
            print("No valid story points found in similar stories, returning None")
            return None
    print("No similar stories found, returning None")
    return None

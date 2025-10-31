import matplotlib.pyplot as plt
import pandas as pd
def show_project_metrics(project_keys):
    all_data = []
    for project_key in project_keys:
      df_train = pd.read_csv(f"./data/TAWOS/{project_key}-train.csv")
      num_stories = len(df_train)
      all_data.append((project_key, num_stories))
    plt.figure(figsize=(10, 6))
    plt.bar([x[0] for x in all_data], [x[1] for x in all_data], color='blue', alpha=0.7)
    plt.title('Number of User Stories per Project')
    plt.xlabel('Project')
    plt.ylabel('Number of User Stories')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(f"./data/comparison/all_projects_num_user_stories.png")
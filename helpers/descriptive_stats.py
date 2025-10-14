import matplotlib.pyplot as plt
import pandas as pd
def show_project_metrics(project_keys= ['ALOY', 'MESOS']):
  # Show the distribution of story points for each project
    all_data = []
    for project_key in project_keys:
      df_train = pd.read_csv(f"./data/TAWOS/{project_key}-train.csv")
      # plt.figure(figsize=(10, 6))
      # plt.hist(df_train['storypoint'], bins=range(0, df_train['storypoint'].max() + 2), alpha=0.7, color='blue', edgecolor='black')
      # plt.title(f'Story Point Distribution for Project {project_key}')
      # plt.xlabel('Story Points')
      # plt.ylabel('Frequency')
      # plt.xticks(range(0, df_train['storypoint'].max() + 2))
      # plt.grid(axis='y', alpha=0.75)
      # plt.show()
      # Save to ./output/
      # plt.savefig(f"./output/{project_key}_storypoint_distribution.png")
      # Per project, show the amount of user stories (1 row is one user story, so count the rows)
      num_stories = len(df_train)
      all_data.append((project_key, num_stories))
    # Show all_data in a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar([x[0] for x in all_data], [x[1] for x in all_data], color='blue', alpha=0.7)
    plt.title('Number of User Stories per Project')
    plt.xlabel('Project')
    plt.ylabel('Number of User Stories')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    # Save to ./output/
    plt.savefig(f"./output/all_projects_num_user_stories.png")
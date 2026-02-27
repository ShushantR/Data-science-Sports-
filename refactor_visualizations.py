import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_toss_match_winner(match_data):
    """
    Beautified plot showing if the toss winner went on to win the match.
    """
    toss = (match_data['toss_winner'] == match_data['winner']).map({True: 'Yes', False: 'No'})

    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    ax = sns.countplot(x=toss, palette='viridis')

    plt.title('Toss Winner is Match Winner', fontweight='bold', fontsize=15)
    plt.xlabel('Toss Winner Wins Match', fontsize=12)
    plt.ylabel('Number of Matches', fontsize=12)

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), 
                    textcoords='offset points')
    plt.show()

def plot_dismissal_kind(df_raina):
    """
    Improved horizontal bar chart for dismissal kinds.
    """
    dismissal_counts = df_raina['dismissal_kind'].value_counts()

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.barplot(x=dismissal_counts.values, y=dismissal_counts.index, palette='magma')

    plt.title("Dismissal Kind Distribution", fontweight='bold', fontsize=15)
    plt.xlabel("Count", fontsize=12)
    plt.ylabel("Dismissal Kind", fontsize=12)
    plt.show()

def plot_top_venues(venue_df):
    """
    Improved horizontal bar chart for top venues.
    """
    top_10_venues = venue_df.head(10)

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.barplot(x='Total', y='Stadium', data=top_10_venues, palette='coolwarm')

    plt.title('Top 10 IPL Venues', fontweight='bold', fontsize=15)
    plt.xlabel('Number of Matches', fontsize=12)
    plt.ylabel('Stadium', fontsize=12)
    plt.show()

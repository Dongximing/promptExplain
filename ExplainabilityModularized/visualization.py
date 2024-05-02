import matplotlib.pyplot as plt
import pandas as pd


# TODO: untested code, test and incorporate to main.py
def plot_average_importance(df, fig_save_path):
    # Columns to visualize
    columns_to_visualize = [col for col in df.columns if '_component_importance' in col]

    # Calculate the average (mean) value of each '_component_importance' column
    average_importance = df[columns_to_visualize].mean()

    # Plotting
    plt.figure(figsize=(10, 6))  # Set the figure size
    average_importance.plot(kind='bar', color='skyblue')  # Create a bar chart
    plt.title('Average Component Importance')  # Chart title
    plt.xlabel('Component')  # X-axis label
    plt.ylabel('Average Importance Value')  # Y-axis label
    plt.xticks(rotation=45, ha="right")  # Rotate the x-axis labels for better readability
    plt.ylim(0, 1)
    plt.tight_layout()  # Adjust the layout to make room for the rotated x-axis labels
    plt.show()
    plt.savefig(fig_save_path)


if __name__ == '__main__':
    # Load the modified DataFrame
    df = pd.read_csv('Results/Patient_0/final_results_100_processed.csv')  # Update the path accordingly
    plot_average_importance(df, fig_save_path='results/temp.png')

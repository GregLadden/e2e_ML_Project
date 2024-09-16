import sys
from src.data_loading import load_housing_data
from src.visualization import (
    plot_histogram,
    plot_histogram_by_proximity,
    plot_ocean_proximity_histogram,
    plot_scatter,
    plot_violin,
    show_correlation_with_ocean_proximity
)

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <command>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create_test_set":
        data = load_housing_data()
        print(data)
    elif command == "plot_histogram":
        plot_histogram()
    elif command == "plot_histogram_by_proximity":
        plot_histogram_by_proximity()
    elif command == "plot_ocean_proximity_histogram":
        plot_ocean_proximity_histogram()
    elif command == "plot_scatter":
        plot_scatter()
    elif command == "plot_violin":
        plot_violin()
    elif command == "show_correlation_with_ocean_proximity":
        show_correlation_with_ocean_proximity()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: create_test_set, plot_histogram, plot_histogram_by_proximity, plot_ocean_proximity_histogram, plot_scatter, plot_violin, show_correlation, show_correlation_with_ocean_proximity")
        sys.exit(1)

if __name__ == "__main__":
    main()

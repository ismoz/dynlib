import dynlib.plot as plot
import numpy as np

# Sample data: a dictionary where keys are categories and values are data arrays
data = {
    'Category A': np.random.randn(100),
    'Category B': np.random.randn(100) + 1,
    'Category C': np.random.randn(100) - 1,
}

# Create facets: 2 columns, with a title
for ax, key in plot.facet.wrap(data.keys(), cols=2, title='Data by Category'):
    values = data[key]
    ax.hist(values, bins=20, alpha=0.7)
    ax.set_title(f'Histogram for {key}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

# Display the plot
plot.export.show()
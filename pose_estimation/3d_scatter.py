from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Create a figure and a 3D axis
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')

# Define the data for the scatter plot
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [5, 6, 2, 3, 13, 4, 1, 2, 4, 8]
z = [2, 3, 3, 3, 5, 7, 9, 11, 9, 10]

# Create the scatter plot
scatter = ax.scatter(x, y, z, c='r', marker='o')

# Remove the grid background
ax.grid(False)

# Add the index of each point as a text on top of each point
for i, txt in enumerate(range(len(x))):
    ax.text(x[i], y[i], z[i], str(txt), color='black')

# Set the labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the plot
plt.show()
import random

def objective_function(x):
    """
    Objective function to maximize: f(x) = x^2
    """
    return x * x

def hill_climbing(starting_point, max_iterations=1000, step_size=0.1):
    """
    Hill climbing algorithm to maximize the objective function.
    """
    # Initialize the current solution with the user-provided starting point
    current_solution = starting_point
    current_value = objective_function(current_solution)
    
    # Start iterating
    for _ in range(max_iterations):
        # Generate a new solution by adding or subtracting a small step
        new_solution = current_solution + random.choice([-1, 1]) * step_size
        
        # Calculate the value of the new solution
        new_value = objective_function(new_solution)
        
        # If the new solution is better, update the current solution
        if new_value > current_value:
            current_solution = new_solution
            current_value = new_value
    
    return current_solution, current_value

# Take user input for the starting point
starting_point = float(input("Enter your starting point: "))

# Perform hill climbing optimization
best_solution, best_value = hill_climbing(starting_point)

print("Best solution:", best_solution)
print("Best value:", best_value)


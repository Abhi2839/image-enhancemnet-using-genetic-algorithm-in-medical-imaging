import cv2
import numpy as np
import random

# Use raw string for correct file path handling
image_path = r'C:\Users\gabhe\OneDrive\Desktop\xry.jpg'

# Load image in grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Image not found or cannot be read.")
    exit()

# **Genetic Algorithm Parameters**
POP_SIZE = 10      # Number of solutions
GENERATIONS = 20   # Iterations
MUTATION_RATE = 0.2  # Mutation probability

# **Step 1: Initialize Population (Gamma values)**
population = [random.uniform(0.5, 2.5) for _ in range(POP_SIZE)]  # Gamma values between 0.5 to 2.5


# **Step 2: Image Enhancement Function (Gamma Correction)**
def gamma_correction(img, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


# **Step 3: Fitness Function (PSNR)**
def psnr(original, enhanced):
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:
        return 100  # Ideal case
    return 10 * np.log10(255 ** 2 / mse)


# **Step 4: Selection - Keep best half based on PSNR**
def selection(population):
    population.sort(key=lambda x: -psnr(image, gamma_correction(image, x)))
    return population[:POP_SIZE // 2]  # Top 50%


# **Step 5: Crossover - Create new candidates**
def crossover(parents):
    children = []
    for _ in range(POP_SIZE - len(parents)):
        p1, p2 = random.sample(parents, 2)
        child = (p1 + p2) / 2  # Average gamma value
        children.append(child)
    return parents + children


# **Step 6: Mutation - Introduce small changes**
def mutate(population):
    for i in range(len(population)):
        if random.random() < MUTATION_RATE:
            population[i] += random.uniform(-0.2, 0.2)  # Small mutation
            population[i] = max(0.5, min(2.5, population[i]))  # Keep in range
    return population


# **Step 7: Run Genetic Algorithm**
for generation in range(GENERATIONS):
    print(f"Generation {generation + 1}")
    parents = selection(population)  # Select best candidates
    offspring = crossover(parents)   # Create new candidates
    population = mutate(offspring)   # Apply mutations

# **Step 8: Get Best Solution**
best_gamma = max(population, key=lambda x: psnr(image, gamma_correction(image, x)))
enhanced_image = gamma_correction(image, best_gamma)

# Display Results 
print(f"Best Gamma: {best_gamma:.2f}")
cv2.imshow('Original Image', cv2.resize(image, (1000, 1000)))  # Resize to fit screen
cv2.imshow('Enhanced Image', cv2.resize(enhanced_image, (1000,1000)))
cv2.waitKey(0)
cv2.destroyAllWindows()
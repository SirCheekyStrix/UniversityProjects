import random
import csv

def generate_points(max_x, max_y, number_of_points):
    return [(random.randint(0, max_x), random.randint(0, max_y)) for _ in range(number_of_points)]

def save_points_to_csv(points, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y"]) #csv headlines
        writer.writerows(points)


max_x = 50
max_y = 100
number_of_points = 1000
points = generate_points(max_x, max_y, number_of_points)

filename = "Points.csv"
save_points_to_csv(points, filename)

print(f"Points have been saved in {filename}")

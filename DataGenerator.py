import random
import csv

def generate_unique_coordinates(existing_coordinates, x_min, x_max, y_min, y_max):
    while True:
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
        if (x, y) not in existing_coordinates:
            existing_coordinates.add((x, y))
            return x, y

def generate_barley_yield(number_of_fields):
    fields = []
    existing_coordinates = set()
    for i in range(number_of_fields):
        value = random.randint(3000, 8000)  #random yield
        x, y = generate_unique_coordinates(existing_coordinates, -1000, 1000, -1000, 1000)
        fields.append({"id": i * 3, "yield": value, "x": x, "y": y})  # IDs as 0, 3, 6, etc.
    return fields

def generate_equal_barley_processed(number_of_breweries):
    breweries = []
    existing_coordinates = set()
    processed_value = random.randint(900, 4000)  #random processed yield
    for i in range(number_of_breweries):
        x, y = generate_unique_coordinates(existing_coordinates, -1000, 1000, -1000, 1000)
        breweries.append({"id": i * 3 + 1, "processed": processed_value, "x": x, "y": y})  # IDs as 1, 4, 7, etc.
    return breweries

def distribute_beer_to_pubs(total_beer, number_of_pubs):
    pubs = []
    existing_coordinates = set()
    remaining_beer = total_beer
    for i in range(number_of_pubs - 1):
        share = random.randint(0, remaining_beer // 2)
        x, y = generate_unique_coordinates(existing_coordinates, -1000, 1000, -1000, 1000)
        pubs.append({"id": i * 3 + 2, "beer": share, "x": x, "y": y})  # IDs as 2, 5, 8, etc.
        remaining_beer -= share
    x, y = generate_unique_coordinates(existing_coordinates, -1000, 1000, -1000, 1000)
    pubs.append({"id": (number_of_pubs - 1) * 3 + 2, "beer": remaining_beer, "x": x, "y": y})
    return pubs

def generate_lanes_yield_to_brewery(fields, breweries):
    lanes = []
    for field in fields:
        number_of_lanes = random.randint(1, len(breweries))
        destinations = random.sample(breweries, number_of_lanes)
        for dest in destinations:
            capacity = random.randint(100, 1000)  #capacity in kg
            lanes.append({
                "from": field["id"],
                "to": dest["id"],
                "capacity": capacity
            })
    return lanes

def generate_lanes_brewery_to_pub(breweries, pubs):
    lanes = []
    for brewery in breweries:
        number_of_lanes = random.randint(1, len(pubs))
        destinations = random.sample(pubs, number_of_lanes)
        for dest in destinations:
            capacity = random.randint(100, 5000)  #capacity in liters
            lanes.append({
                "from": brewery["id"],
                "to": dest["id"],
                "capacity": capacity
            })
    return lanes

def save_all_to_csv(filename, fields, breweries, pubs, lanes):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Category", "ID", "Yield (kg)", "Processed (kg)", "Beer (liters)", 
            "X Coordinate", "Y Coordinate", "Lane From", "Lane To", "Capacity (kg/liters)"
        ])
        for field in fields:
            writer.writerow([
                "Field", field["id"], field["yield"], "", "",
                field["x"], field["y"], "", "", ""
            ])
        for brewery in breweries:
            writer.writerow([
                "Brewery", brewery["id"], "", brewery["processed"], "",
                brewery["x"], brewery["y"], "", "", ""
            ])
        for pub in pubs:
            writer.writerow([
                "Pub", pub["id"], "", "", pub["beer"],
                pub["x"], pub["y"], "", "", ""
            ])
        for lane in lanes:
            writer.writerow([
                "Lane", "", "", "", "",
                "", "", lane["from"], lane["to"], lane["capacity"]
            ])

number_of_fields = int(input("Enter the number of fields: "))
fields = generate_barley_yield(number_of_fields)

number_of_breweries = int(input("Enter the number of breweries: "))
breweries = generate_equal_barley_processed(number_of_breweries)

total_beer = sum([brewery["processed"] * random.randint(1, 10) for brewery in breweries])

number_of_pubs = int(input("Enter the number of pubs: "))
pubs = distribute_beer_to_pubs(total_beer, number_of_pubs)

lanes_field_to_brewery = generate_lanes_yield_to_brewery(fields, breweries)
lanes_brewery_to_pub = generate_lanes_brewery_to_pub(breweries, pubs)

combined_lanes = lanes_field_to_brewery + lanes_brewery_to_pub

save_all_to_csv("data.csv", fields, breweries, pubs, combined_lanes)
print("Data have been saved to 'data.csv'")

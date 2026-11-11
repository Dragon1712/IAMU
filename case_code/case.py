import random

case_num = 20

for case_id in range(case_num):

    # case information
    b = 5
    crane_num = 30
    num_ships = 50 * b

    # Generating ship information
    ships = []
    for _ in range(num_ships):
        arrival_time = round(random.uniform(0, 15 * 24 * b), 2)
        cargo_capacity = random.randint(32, 480)
        vessel_length = random.randint(150, 400)
        max_crane = vessel_length // 24
        ships.append([arrival_time, cargo_capacity, vessel_length, max_crane])

    # Sort by arrival time
    ships.sort(key=lambda x: x[0])

    # Writing to a file
    file_name = f'./test/test_{case_id}.txt'
    first_line = True
    with open(file_name, 'w',encoding='utf-8') as file:
        for ship in ships:
            if first_line:
                first_line = False
                file.write('\t'.join(map(str, ship + [crane_num])))
            else:
                file.write('\n'+'\t'.join(map(str, ship)))
        file.flush()
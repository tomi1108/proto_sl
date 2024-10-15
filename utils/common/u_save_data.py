import csv

def save_data(
    path: str,
    write_data: list
):
    
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(write_data)
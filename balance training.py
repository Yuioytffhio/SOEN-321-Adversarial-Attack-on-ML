import csv

input_file="data/KDDTest+.txt"
output_file="data/KDDTest+_balanced.txt"
target_column=41
value1="normal"
max_not_value1_count=9711


def copy_until_value1_reaches_x():
    count_not_val1 = 0

    with open(input_file, newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header)

        if isinstance(target_column, str):
            col_idx = header.index(target_column)
        else:
            col_idx = target_column

        for row in reader:
            val = row[col_idx]

            if val != value1:
                if count_not_val1 < max_not_value1_count:
                    writer.writerow(row)
                    count_not_val1 += 1
                else:
                    continue
            else:
                writer.writerow(row)

    print(f"Finished. Attack count reached {count_not_val1} (limit was {max_not_value1_count}).")


copy_until_value1_reaches_x()
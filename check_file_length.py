import os
import time


if __name__ == "__main__":

    msa_data_path = "/home/mnowak/care/care-output/"
    msa_file_name = "humanchr1430covMSA_1"
    msa_file = os.path.join(msa_data_path, msa_file_name)

    file_ = open(msa_file, "r")

    lines = [line.replace('\n', '') for line in file_]
    print(f"The file has a total of {len(lines)} lines")

    reading = True
    header_line_number = 0
    header_line_numbers = []
    million_counter = 0

    start = time.time()

    while reading:

        if header_line_number >= len(lines):
            reading = False
            continue
            
        header_line_numbers.append(header_line_number)
        if header_line_number >= million_counter*1000000:
            print(f"Currently at line {header_line_number}")
            million_counter += 1

        number_rows, number_columns, anchor_in_msa, anchor_in_file = [int(i) for i in lines[header_line_number].split(" ")]
        header_line_number += number_rows + 1


    end = time.time()
    duration = end - start

    print(f"There were {len(header_line_numbers)} many MSAs found. The process took {duration} seconds")
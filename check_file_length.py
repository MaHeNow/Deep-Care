import os


if __name__ == "__main__":

    msa_data_path = "/home/mnowak/care/care-output/"
    msa_file_name = "humanchr1430covMSA_1"
    msa_file = os.path.join(msa_data_path, msa_file_name)

    number_of_lines = len(open(msa_file).readlines())
    print(number_of_lines)
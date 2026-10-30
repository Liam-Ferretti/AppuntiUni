input_file = "input.txt"
output_file = "output.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Keep everything after the first '$'
        if "$" in line:
            cleaned_line = line.split("$", 1)[1]
            outfile.write(cleaned_line + "\n")

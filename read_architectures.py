
def get_class_label(mapping="class_labels.txt"):
    label_map = {}
    f = open(mapping,'r')
    for line in f:
        line = line.rstrip()
        x = line.split(',')
        if x[0] != "Label":
            label_map[int(x[0])] = x[1]
    return label_map


def read_architectures(architectures_file):
    print("Parsing architectures from "+architectures_file+"...")
    arch_dict = {}
    f = open(architectures_file,'r')
    for line in f:
        # Ignor comments
        if line[0] != '#':
            # Remove trailing newline
            line = line.rstrip()
            x = line.split(',')
            arch_dict[x[0]] = [i for i in x[1:]]
    f.close()
    return arch_dict
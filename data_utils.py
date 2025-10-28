def load_species_names(label_file):
    """
    Reads a Train.txt file and returns a list of species names, 
    where the index corresponds to the class number.
    """
    species_dict = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                filename, class_id = parts
                class_id = int(class_id)
                # Extract species name (everything before first underscore)
                species_name = "_".join(filename.split("_")[:-1])
                # Clean up name: remove numbers and extensions
                species_name = species_name.rsplit("_", 1)[0]
                species_dict[class_id] = species_name.replace("_", " ")
    # Sort by class index and return list
    species_names = [species_dict[i] for i in sorted(species_dict.keys())]
    return species_names


def get_class_names(num_classes):
    # This should be replaced with your actual class name list
    return [f"Bird_Class_{i+1}" for i in range(num_classes)]


def extract_class_names_from_dataset(dataset):
    """
    Extracts species names (class names) from a BirdDataset object.
    Returns a list where index = class number, value = species name.
    """
    mapping = {}
    for _, row in dataset.data.iterrows():
        filename, class_id = row['file_path'], int(row['class'])
        # Get species part only (remove image number + .jpg)
        species_name = "_".join(filename.split("_")[:-1])  # remove trailing numeric part
        species_name = species_name.replace("_", " ")      # make it readable
        mapping[class_id] = species_name
    
    # Sort by class index
    class_names = [mapping[i] for i in sorted(mapping.keys())]
    return class_names
def multiple_find_and_replace(file_path, search_texts, replace_texts):
    if len(search_texts) != len(replace_texts):
        print("Error: search_texts and replace_texts must have the same length.")
        return

    # Read the contents of the file
    with open(file_path, 'r') as file:
        file_contents = file.read()

    # Replace each target string
    for search_text, replace_text in zip(search_texts, replace_texts):
        file_contents = file_contents.replace(search_text, replace_text)

    # Write the new contents back to the file
    with open(file_path, 'w') as file:
        file.write(file_contents)

    print(f"Replaced {len(search_texts)} text(s) in '{file_path}'")

# Example usage
y = []

for x in y:
    file_path = "Environment/objects_assets/" + x + ".xml"
    search_texts = ['name="model"', 'file="', 'material_0','mesh="model"','name="model_collision','mesh="model_collision','"texture"','<body name="'+x+'">']
    replace_texts = ['name="'+x+'"', 'file="../../Environment/objects_assets/'+x+"/", 'material_0_'+x, 'mesh="' + x + '"','name="model_collision_' + x,'mesh="model_collision_' + x,'"texture_'+x+'"','<body name="'+x+'" pos="1.1 -1.5 1"> <freejoint/>']

    multiple_find_and_replace(file_path, search_texts, replace_texts)

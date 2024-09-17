from openai import OpenAI
import Classes
import json
import numpy as np

module_id = "GPT"
import Utils

def get_api_key(filepath="GPT/key.txt"):
    with open(filepath, 'r') as file:
        return file.read().strip()

client = OpenAI(api_key=get_api_key())

def call_gpt_with_json(json_objects,  instruction, json_relations=None, model="gpt-4o"):
    
    if ".json" in json_objects:
        with open(json_objects, 'r') as file:
            json_objects_data = json.load(file)
    else:
        json_objects_data = json_objects

    if json_relations is None:
        json_relations_data = {}
    else:
        if isinstance(json_relations, str):
            with open(json_relations, 'r') as file:
                json_relations_data = json.load(file)
        else:
            json_relations_data = json_relations 
            
    main_prompt = (
        "You are a home robot that is used to carry out home tasks."
        "The actions you have are {pick(x),place(y)}. "
        "You can only pick one item at a time and everything that you pick or place must be in teh json file."
        "Pick would go to object x and pick it. Place would go to object y (where you wanna place the object x that you grasped) and ungrasp it. "
        "I will give you tasks and want you to give me steps of action to do. "
        "If I ask you for an object that isn't available, assume it is the object with the closest resemblance to it. "
        "Give me the steps in this format {action_type, object_name, object_center} and in order. "
        f"Here is the JSON data: {json.dumps(json_objects_data)}"
        f"Here is another JSON data with relations between the objects which may or may not be empty: {json.dumps(json_relations_data)}"
        "Only output the json text, I will be parsing the raw output you give so ensure it's right."
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": main_prompt},
        {"role": "user", "content": instruction}
    ]

    response = client.chat.completions.create(model=model,
    messages=messages)

    output = response.choices[0].message.content

    return output

def parse_output(self,output):
    actions_parsed = []
    
    # Strip out non-JSON parts
    json_start = output.find("[")
    json_end = output.rfind("]") + 1
    json_str = output[json_start:json_end]
    
    try:
        # Load the entire output as a JSON list
        json_data = json.loads(json_str)

        for i, action in enumerate(json_data):
            object_name = action.get("object_name").lower()
            action_type = action.get("action_type").lower()
            x, y, z = action.get("object_center", [None, None, None])
            if x is not None and y is not None and z is not None:
                object = Classes.Object(object_name,np.array([x,y,z]))
                actions_parsed.append(Classes.Action(action_type,object))
            Utils.print_debug(f"Action {i}: {action_type} - {object_name} - {[x,y,z]}",self.args.debug,module_id)
            
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    
    return actions_parsed


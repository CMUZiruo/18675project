import openai
import json
def load_instruction_file(fpath):
    # Open the file for reading
    try:
        with open(fpath, "r") as file:
            # Read the contents of the file
            file_contents = file.read()
            # Process or print the file contents
            return file_contents
    except FileNotFoundError:
        print(f"File not found: {fpath}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def llm_call(messages, temperature=0.2):

    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        temperature=temperature,
                        messages=messages)
    return response

if __name__ == "__main__":
    motion_descriptor = load_instruction_file("motion_descriptor.txt")
    reward_coder = load_instruction_file("reward_coder.txt")

    # Open the JSON file in read mode
    with open("functions.json", "r") as json_file:
        # Parse the JSON data
        fc_json = json.load(json_file)

    # Now 'data' contains the parsed JSON data as a Python dictionary or list
    fc_rewards = fc_json["rewards"]
    fc_constraints = fc_json["constraints"]
    print("Successfully OpenAI key authorization")
import openai
import os
import base64
import ast


# Set up your OpenAI API key
# export OPENAI_API_KEY="sk-your-api-key-here"
openai.api_key = os.getenv('OPENAI_API_KEY')

def prompt_generator(instruction, sample_num=5):
    prompt = f"""You are a generalist agent tasked with controlling a physical robot arm equipped with a two-finger gripper, under natural language supervision from humans. Your objective is to compute and return {sample_num} candidate actions that the robot can take at the current scene, considering a specific instruction and a visual context.
    
    INSTRUCTION: {instruction}
    
    OUTPUT FORMAT: You should output the actions represented in language in a Python list format. For example, ["move forward a little", "rotate arm" ...]. The descriptions should not include object specifics but should focus solely on directional and operational commands relevant to the robot's mechanics.
    
    EXAMPLE of ACTIONS: Actions may include but are not limited to: 'Move forward', 'Move backward a little', 'Move left', 'Roll arm a little', 'Pitch down', 'Rotate gripper 90 degrees counterclockwise'. Ensure actions are calculated with respect to the robot's current position and orientation relative to the object of interest."""

    return prompt

def encode_image(image_path):
    """
    Encode image to base64 string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def action_sampler(prompt, image_path, model="gpt-4-vision-preview"):
    """
    Get a response from GPT model with image input
    """
    try:
        base64_image = encode_image(image_path)
        client = openai.OpenAI()  # Create client instance
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return extract_list_from_response(response.choices[0].message.content)
    except Exception as e:
        return f"An error occurred: {str(e)}"


def extract_list_from_response(response_text):
    """
    Extract the list from a text response that might contain additional content
    """
    try:
        # Find the first '[' and last ']'
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            # Extract just the list part
            list_str = response_text[start_idx:end_idx + 1]
            # Convert to actual list
            return ast.literal_eval(list_str)
        else:
            print("No list found in response")
            return None
    except Exception as e:
        print(f"Error parsing response: {e}")
        print("Raw response:", response_text)
        return None

def main():
    # gpt_model = "gpt-4o-mini"
    gpt_model = "gpt-4o"
    # Example usage with an image
    image_path = "./test.png"  # Replace with your image path
    instruction = "pick the blue cup"
    prompt = prompt_generator(instruction)
    actions = action_sampler(prompt, image_path, gpt_model)
    print(f"Prompt: {prompt}")
    print(f"Actions: {actions}")

if __name__ == "__main__":
    main()
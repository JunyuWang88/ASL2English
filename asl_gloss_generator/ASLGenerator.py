import openai
import os
# Make sure to replace 'your_api_key' with your actual API key
openai.api_key = ""

# Get the current directory of the script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Replace the English sentences in the list with the ones you want to translate
english_sentences = []
with open(os.path.join(current_directory, "english_gloss.txt"), "r") as file:
    for line in file:
        english_sentences.append(line.strip())


def translate_to_asl(chuck_of_english_sentences):
    prompt = f"Translate the user's English sentence input into an ASL gloss sequence, adhering to ASL grammar rules. " \
             f"The output should only consist of the gloss sequence. Here are some examples:\nInput: I need to drink " \
             f"water. Output: WATER DRINK I NEED.\nInput: What is your name? Output: YOUR NAME WHAT?\nInput: Ava is " \
             f"58. Output: AVA AGE 58.\nInput: Bob ate 6 bananas. Output: 6 BANANA BOB EAT.\nInput: Nice to meet you. " \
             f"Output: NICE MEET YOU.\nInput: It's a nice day. Output: NICE DAY.\nInput: This is sign language. " \
             f"Output: LANGUAGE SIGN THIS.\nInput: My name is Ava. Output: MY NAME AVA.\n Please only return the " \
             f"translated ASL result seperated by \n. Translate the following English sentences to ASL:  {chuck_of_english_sentences} "
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=2000,
    )
    asl_gloss_sequence = response.choices[0].text.strip()
    return asl_gloss_sequence


# Split the sentences into groups of 10
chunks = [english_sentences[i:i + 50] for i in range(0, len(english_sentences), 50)]
total_chunks = len(chunks)
# Translate each group and concatenate the results
asl_gloss_sequences = []

for idx, chunk in enumerate(chunks):
    asl_gloss_sequence = translate_to_asl(chunk)
    asl_gloss_sequences.append(asl_gloss_sequence)
    progress_percentage = (idx + 1) / total_chunks * 100
    print(f"Progress: {progress_percentage:.2f}%")


# Save the asl_gloss_sequences to a local file
with open("asl_gloss_sequences.txt", "w") as file:
    for gloss in asl_gloss_sequences:
        file.write(f"{gloss}\n")

print("Output saved to 'asl_gloss_sequences.txt'")


import os
import openai

# Replace 'your_api_key' with your actual OpenAI API key
api_key = ''

openai.api_key = api_key

def translate_to_asl_gloss(text):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Please translate the following English text to American Sign Language (ASL) gloss, adhering to ASL grammar rules.:\n\n{text}\n\nASL gloss translation:",
            temperature=0.5,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        print(response)
    except Exception as e:
        print(e)


    asl_gloss = response.choices[0].text.strip()
    return asl_gloss

english_paragraph = "what is your name"

asl_translation = translate_to_asl_gloss(english_paragraph)
print("ASL gloss translation:", asl_translation)
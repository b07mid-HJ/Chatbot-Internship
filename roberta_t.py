import json
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

# Create the question answering pipeline
nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Define a function to read the context from a file
def read_context(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    context, content = text.split(":", maxsplit=1)
    return context.strip(), content.strip()

# Define a function to get the answer to a question based on a context
def get_answer(context, question):
    result = nlp(question=question, context=context)
    answer = result["answer"] if result["score"] > 0.5 else "No answer found"
    return answer

# Specify the folder path where the files are located
folder_path = "/content/files"

# Get the list of file names in the folder
file_names = os.listdir(folder_path)
question = input("Enter your question: ")

# Flag to track if an answer is found
answer_found = False
with open("/content/intents.json", "r") as file:
    intents_data = json.load(file)
# Get the list of existing intents
    intents = intents_data
# Process each file and check for an answer
for file_name in file_names:
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        context, content = read_context(file_path)
        result = nlp(question=question, context=content)
    
    # Check if an answer was found
        if result["score"] > 0.5:
            answer_found = True
        
            intent = {
                "tag": context,
                "patterns": [question],
                "responses": [result["answer"]],
            }
        
        # Add the intent to the list
            intents["intents"].append(intent)
intents_data = {
    "intents": intents
}
# If no answer was found
with open("/content/intents.json", "w") as file:
    json.dump(intents, file, indent=4)
if not answer_found:
    print("No file found with an answer to the question.")
else:
    print(result["answer"])
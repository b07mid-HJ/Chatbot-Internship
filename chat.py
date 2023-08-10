import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.8:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
    # Your previous code to add the intent to intents.json

      import random
      import json
      import os

      from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

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
      question = " ".join(sentence)  # Assuming 'sentence' contains the user's question

      # Flag to track if an answer is found
      answer_found = False
      with open("/content/intents.json", "r") as file:
          intents_data = json.load(file)
      intents = intents_data["intents"]  # Get the list of existing intents

      # Check if the intent already exists in the intents list
      def intent_exists(intent):
          for existing_intent in intents:
              if existing_intent["tag"] == intent["tag"] and intent["patterns"][0] in existing_intent["patterns"]:
                  return True
          return False

      # Process each file and check for an answer
      for file_name in file_names:
          if file_name.endswith(".txt"):
              file_path = os.path.join(folder_path, file_name)
              context, content = read_context(file_path)
              result = nlp(question=question, context=content)

              # Check if an answer was found
              if result["score"] > 0.5:
                  answer_found = True
                  res=result["score"]
                  intent = {
                      "tag": context,
                      "patterns": [question],
                      "responses": [result["answer"]],
                  }

                  # Check if the intent already exists before adding it
                  if not intent_exists(intent):
                      intents.append(intent)

      intents_data["intents"] = intents  # Update the intents data

      # Write the updated intents back to the JSON file
      with open("/content/intents.json", "w") as file:
          json.dump(intents_data, file, indent=4)

      if not answer_found:
          print("No file found with an answer to the question.")
      else:
          print(res)

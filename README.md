# IMPORTANT
A lot of the code to create the model was sourced from [this video](https://www.youtube.com/watch?v=1lwddP0KUEg) and the website below:
https://towardsdatascience.com/how-to-create-a-chatbot-with-python-deep-learning-in-less-than-an-hour-56a063bdfc44 


## JSON
The JSON file should be formatted like this to add new content:
```json
    {"tag": "NAME OF THE CATEGORY OF RESPONSE",
        "patterns": [TYPES OF INPUTS],
        "responses": [BOT ANSWERS]
    },
```
Sample:
```json
    {"tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Until next time"],
        "responses": ["It was nice talking with you!", "Bye!", "Goodbye!"]
```

## Training.py
This is the file to train the model. Everytime the JSON is updated, this file should be run to generate the new and current model.

## Chatbot.py
Main file that produces tangible interactions. Integrate with Discord here.

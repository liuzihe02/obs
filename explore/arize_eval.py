# %% Load in imports
import pandas as pd
from datasets import load_dataset
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR

# %% prepare dataset

qa_data = load_dataset("json", data_files=str(DATA_DIR / "qa_data.json"))

# Convert the huggingface data to pandas
df = qa_data["train"].to_pandas()

# at this point, this is in "knowledge, question, right_answer, hallucinated answer"
# we will take the first n rows,

# %% Prepare dataset

df = pd.DataFrame(
    [
        {
            "reference": "The Eiffel Tower is located in Paris, France. It was constructed in 1889 as the entrance arch to the 1889 World's Fair.",
            "query": "Where is the Eiffel Tower located?",
            "response": "The Eiffel Tower is located in Paris, France.",
        },
        {
            "reference": "The Great Wall of China is over 13,000 miles long. It was built over many centuries by various Chinese dynasties to protect against nomadic invasions.",
            "query": "How long is the Great Wall of China?",
            "response": "The Great Wall of China is approximately 13,171 miles (21,196 kilometers) long.",
        },
        {
            "reference": "The Amazon rainforest is the largest tropical rainforest in the world. It covers much of northwestern Brazil and extends into Colombia, Peru and other South American countries.",
            "query": "What is the largest tropical rainforest?",
            "response": "The Amazon rainforest is the largest tropical rainforest in the world. It is home to the largest number of plant and animal species in the world.",
        },
        {
            "reference": "Mount Everest is the highest mountain on Earth. It is located in the Mahalangur Himal sub-range of the Himalayas, straddling the border between Nepal and Tibet.",
            "query": "Which is the highest mountain on Earth?",
            "response": "Mount Everest, standing at 29,029 feet (8,848 meters), is the highest mountain on Earth.",
        },
        {
            "reference": "The Nile is the longest river in the world. It flows northward through northeastern Africa for approximately 6,650 km (4,132 miles) from its most distant source in Burundi to the Mediterranean Sea.",
            "query": "What is the longest river in the world?",
            "response": "The Nile River, at 6,650 kilometers (4,132 miles), is the longest river in the world.",
        },
        {
            "reference": "The Mona Lisa was painted by Leonardo da Vinci. It is considered an archetypal masterpiece of the Italian Renaissance and has been described as 'the best known, the most visited, the most written about, the most sung about, the most parodied work of art in the world'.",
            "query": "Who painted the Mona Lisa?",
            "response": "The Mona Lisa was painted by the Italian Renaissance artist Leonardo da Vinci.",
        },
        {
            "reference": "The human body has 206 bones. These bones provide structure, protect organs, anchor muscles, and store calcium.",
            "query": "How many bones are in the human body?",
            "response": "The adult human body typically has 256 bones.",
        },
        {
            "reference": "Jupiter is the largest planet in our solar system. It is a gas giant with a mass more than two and a half times that of all the other planets in the solar system combined.",
            "query": "Which planet is the largest in our solar system?",
            "response": "Jupiter is the largest planet in our solar system.",
        },
        {
            "reference": "William Shakespeare wrote 'Romeo and Juliet'. It is a tragedy about two young star-crossed lovers whose deaths ultimately reconcile their feuding families.",
            "query": "Who wrote 'Romeo and Juliet'?",
            "response": "The play 'Romeo and Juliet' was written by William Shakespeare.",
        },
        {
            "reference": "The first moon landing occurred in 1969. On July 20, 1969, American astronauts Neil Armstrong and Edwin 'Buzz' Aldrin became the first humans to land on the moon as part of the Apollo 11 mission.",
            "query": "When did the first moon landing occur?",
            "response": "The first moon landing took place on July 20, 1969.",
        },
    ]
)
df.head()

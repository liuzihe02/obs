import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create generation config with logprobs enabled
generation_config = {
    "temperature": 0.7,
    "response_logprobs": True,  # Enable logprobs in response
    "logprobs": 5,  # Get top 5 logprobs for each token (range: 1-5)
}

# Initialize model with logprobs config
model = genai.GenerativeModel("gemini-1.5-flash-002")

# Generate content with logprobs
response = model.generate_content(
    "Explain how AI works", generation_config=generation_config
)

# Print the response text
print("Generated text:", response.text)

# Access logprobs from the response
if hasattr(response.candidates[0], "logprobs_result"):
    logprobs_result = response.candidates[0].logprobs_result

    print("\nToken probabilities:")
    # Print top candidates at each decoding step
    for i, top_cands in enumerate(logprobs_result.top_candidates):
        print(f"\nStep {i + 1}:")
        for candidate in top_cands.candidates:
            print(f"Token: {candidate.token}")
            print(f"Log Probability: {candidate.log_probability}")

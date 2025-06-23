import json
import evaluate
import nltk
import os
from dotenv import load_dotenv

from openai import AzureOpenAI,OpenAI
import google.generativeai as genai
import boto3
from botocore.exceptions import ClientError

from textwrap import dedent
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM

# load_dotenv(verbose=True)

# --- Download NLTK data for METEOR if not already present ---
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
    print("NLTK data (wordnet and punkt) already present.")
except Exception as e:
    print(f"Error checking NLTK data: {e}. Attempting to download...")
    try:
        nltk.download('wordnet')
        nltk.download('punkt')
        print("NLTK data (wordnet and punkt) downloaded successfully.")
    except Exception as download_e:
        print(f"Failed to download NLTK data: {download_e}")
        print("WARNING: METEOR metric might not function correctly without 'wordnet' and 'punkt' resources.")


# --- Load the Evaluation Metrics ---
print("Loading evaluation metrics...")
try:
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    print("All core metrics loaded.")
except Exception as e:
    print(f"Error loading metrics: {e}")
    print("Please ensure you have all necessary libraries installed (e.g., pip install evaluate rouge_score sacrebleu nltk) and internet connection.")
    exit()

# --- Dummy Data ---
# prompt = "Explain the concept of photosynthesis simply."
# reference_answer = "Photosynthesis is the process used by plants to convert light energy into chemical energy, using water and carbon dioxide to create sugars and oxygen."
# llm_response = "Plants use sunlight to make food. They take in water and carbon dioxide, and release oxygen. This process is called photosynthesis."

# prompt = """Read the following patient note and provide a brief summary of the patient's chief complaints and current symptoms. Focus on key medical terms and duration if mentioned.

# Patient Note:'Patient, Ms. Clara Barton, 68 years old, reports increasing dyspnea on exertion over the last month. She also mentions intermittent chest tightness, especially with physical activity, which she describes as a dull pressure. Furthermore, she has experienced occasional, non-productive coughs in the mornings for approximately two weeks. No fever or chills reported.'"""


# llm_response= "Ms. Barton experiences breathlessness during physical activity for a month, has periodic chest pressure when active, and a dry cough in the mornings for about two weeks."


# reference_answer = "Patient reports worsening dyspnea on exertion for one month, intermittent exertional chest tightness, and a non-productive morning cough for two weeks."

# print(f"\n--- Evaluation Inputs ---")
# print(f"Prompt: {prompt}")
# print(f"Reference Answer: '{reference_answer}'")
# print(f"LLM Response: '{llm_response}'\n")

# --- Function to calculate requested scores ---
endpoint = "https://omnicaaihub1774838731.openai.azure.com/"
deployment = "omnica-gpt-4o"
api_key = "w6qM5xlLw6aeFHIrqBjW8u7oTlsI7zdcWaQGEwLShRapIlWTjq4oJQQJ99BDACHYHv6XJ3w3AAAAACOGW6Nj"
api_version = "2024-12-01-preview"


client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
)

client1= "client1"
genai.configure(api_key="AIzaSyBKKENthQ2WS5DvjNCC2SOYTjezMUlaw14")
model1=genai.GenerativeModel("gemini-1.5-flash")

YOUR_API_KEY = "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"
client2 = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")
model2="sonar-pro"

# Create a Bedrock Runtime client in the AWS Region you want to use.
client3 = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "amazon.nova-lite-v1:0"

with open("prompt_context.json", "r", encoding="utf-8") as f:
    prompt_data = json.load(f)

def format_prompt(question: str, context: str):
    return dedent(
        f"""
        Use the following context:
        ```
        {context}
        ```
        to answer the question:
        ```
        {question}
        ```

        Your answer must be succinct!
        Answer:
    """
    )

def predict(prompt, client, model):
    # Sends the evaluation prompt to the specified LLM client and model to get a judgment on other two LLM responses.
   
    if client==client1:
        prompt_text = prompt
        chat = model.start_chat()
        response = chat.send_message(prompt_text)
        return response.text.strip()
    
    elif client==client3:
        conversations = [
        {"role": "user", "content": [{"text":prompt}]}
        ]
        try:
            response = client.converse(
                modelId=model,
                messages=conversations,
                inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
            )

            response_text = response["output"]["message"]["content"][0]["text"]
            return response_text

        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
            exit(1)
    messages = [
    {"role": "system", "content": "You are a fair and knowledgeable LLM evaluator."},
    {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model,  
        messages=messages,
        temperature=1.0
    )
    return response.choices[0].message.content.strip()

@dataclass
class QuestionAnswer:
    question: str
    reference_answer: str
    context: str
    answer:List[str]

def extract_predictions(data,models) -> QuestionAnswer:
    answers=[]
    for model in models:
        answers.append(predict(format_prompt(data["prompt"], data["context"]),model[1], model[2]))
       
    return QuestionAnswer(
            question=data["prompt"],
            reference_answer=data["reference_answer"],
            context=data["context"],
            answer=answers,
        )
        


    
class AzureCriticModel(DeepEvalBaseLLM):

    def __init__(self, openai_client_instance: AzureOpenAI, deployment_name: str):
        """
        Initializes the custom model with an AzureOpenAI client and its deployment name.

        Args:
            openai_client_instance: An instantiated openai.AzureOpenAI client.
            deployment_name: The name of the specific Azure deployment to use (e.g., "omnica-gpt-4o").
        """
        self.client = openai_client_instance  # Store the actual client instance
        self.deployment_name = deployment_name # Store the deployment name

    def load_model(self):
        """
        Returns the instantiated AzureOpenAI client.
        This method is required by DeepEvalBaseLLM.
        """
        return self.client

    def generate(self, prompt: str) -> str:
        """
        Synchronously generates a response using the Azure OpenAI chat completion.
        """
        return predict(prompt,self.client,self.deployment_name)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        """
        Returns the deployment name, which DeepEval uses for reporting.
        """
        return self.deployment_name
    
critic_model=AzureCriticModel(openai_client_instance=client,deployment_name=deployment )


def calculate_metrics_for_generative_text(llm_response_str, reference_answer_str,test_case):
    
    scores={}
    # scores["Exact_Match_Accuracy"] = 1.0 if llm_response_str.strip() == reference_answer_str.strip() else 0.0

    # 2. ROUGE and F1-score (ROUGE-1 F1)
    # ROUGE provides F1, Precision, and Recall for each type (rouge1, rouge2, rougeL, rougeLsum)
    try:
        rouge_results = rouge_metric.compute(
            predictions=[llm_response_str],
            references=[reference_answer_str]
        )
        scores["ROUGE-1_F1"] = round(rouge_results["rouge1"], 4) # This is the F1 score for unigram overlap
        scores["ROUGE-2_F1"] = round(rouge_results["rouge2"], 4)
        scores["ROUGE-L_F1"] = round(rouge_results["rougeL"], 4)
    except Exception as e:
        scores["ROUGE_Scores"] = f"Error: {e}"

    # 3. BLEU Score
    try:
        # BLEU expects references as a list of lists (even if only one reference)
        # and predictions as a list of strings
        bleu_results = bleu_metric.compute(
            predictions=[llm_response_str],
            references=[[reference_answer_str]]
        )
        scores["BLEU"] = round(bleu_results["bleu"], 4)
    except Exception as e:
        scores["BLEU"] = f"Error: {e}"

    # 4. METEOR Score
    try:
        # METEOR expects references as a list of lists (even if only one reference)
        # and predictions as a list of strings
        meteor_results = meteor_metric.compute(
            predictions=[llm_response_str],
            references=[[reference_answer_str]]
        )
        scores["METEOR"] = round(meteor_results["meteor"], 4)
    except Exception as e:
        scores["METEOR"] = f"Error: {e}"

    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully.")
    print("\nGenerating embeddings for texts...")
    embedding_llm_response = model.encode(llm_response_str)
    embedding_reference = model.encode(reference_answer_str)

    # Reshape embeddings to 2D arrays for sklearn's cosine_similarity
    embedding_llm_response_reshaped = embedding_llm_response.reshape(1, -1)
    embedding_reference_reshaped = embedding_reference.reshape(1, -1)

    similarity_score = cosine_similarity(embedding_llm_response_reshaped, embedding_reference_reshaped)[0][0]
    # --- Execute Evaluation and Print JSON Output ---

    scores["Cosine_Similarity_Score"]=round(float(similarity_score), 4)

    coherence_metric = GEval(
        name="Coherence",
        criteria="Coherence - the collective quality of all sentences in the actual output",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=critic_model
    )

    coherence_metric.measure(test_case)
    scores["Coherence_Score"]=coherence_metric.score
    # final_scores["Coherence_reason"]=coherence_metric.reason

    metric1 = HallucinationMetric(threshold=0.5,model=critic_model)
    metric1.measure(test_case)
    scores["Hallucination_Score"]=metric1.score

    metric2 = AnswerRelevancyMetric(threshold=0.5,model=critic_model)
    metric2.measure(test_case)
    scores["Relevance_Score"]=metric2.score

    metric3 = FaithfulnessMetric(threshold=0.5,model=critic_model)
    metric3.measure(test_case)
    scores["Faithfulness_Score"]=metric3.score
    return scores

models = [["Perplexity",client2,model2], ["Gemini",client1,model1], ["OpenAI",client,deployment], ["AWS Nova-Lite",client3,model_id]]

rows: List[Dict] = []
for i, item in enumerate(prompt_data):
    print(f"Evaluating prompt {i+1}/{len(prompt_data)}...")
    prediction=extract_predictions(item,models)
    response={
        "SNO":i+1,
        "prompt":item["prompt"],
        "models":{}
    }
    k=0
    for res in prediction.answer:
        test_case = LLMTestCase(input=item["prompt"], actual_output=res,retrieval_context=[prediction.context], context=[prediction.context])
        scores=calculate_metrics_for_generative_text(res,prediction.reference_answer,test_case)
        response["models"][models[k][0]]=scores
        k+=1
    rows.append(response)

with open("multi_model_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)

print("Completed evaluating scores......")
#  uvicorn main:app --reload --port 8000

import torch
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from typing import Tuple

from helper import LSTMToolkit, reverse_label_encoding


class ReviewInput(BaseModel):
    review: str = Field(min_length=1)


app = FastAPI(
    title="Movie Review Classifier",
    description="Predicts the score of the movie review"
)

# Mount static files
app.mount("/static", StaticFiles(directory="../static"), name="static")

# Set up template rendering
templates = Jinja2Templates(directory="../templates")

# Load model
LSTM_TOOLKIT = LSTMToolkit()
MODEL = LSTM_TOOLKIT.model
MODEL.load_state_dict(torch.load("../classifier/lstm_sentiment_classifier.pt", weights_only=True))
MODEL.eval()

# Define LLM prompt
LLM_NAME = "llama3.1"
LLM = ChatOllama(model=LLM_NAME, temperature=0.0)
TEMPLATE = """
You are a critic that reviews other reviews.
Generate a brief humorous response based ONLY on the {predicted_review_score} and {review}. 
"""
PROMPT_TEMPLATE = ChatPromptTemplate.from_template(TEMPLATE)

@app.get("/", response_class=HTMLResponse)
async def chat_get(request: Request) -> HTMLResponse:
    """
    Handle GET requests for the chat page. Renders the
    initial page where the user can enter their message.
    :param request: The incoming request object.
    :return: HTMLResponse: Renders the chat page template with
             user input and model response.
    """
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_input": None,
        "predicted_score": None,
        "llm_response": None
    })


@app.post("/", response_class=HTMLResponse)
async def chat_post(request: Request, user_input: str = Form(...)) -> HTMLResponse:
    """
    Handle POST requests when a user submits their message. This function
    processes the user's input, generates a response using the model, and
    renders the updated page with the user input and model reply.
    :param request: The incoming request object.
    :param user_input: The message submitted by the user via the form.
    :return: HTMLResponse: Renders the chat page with the user's input and the corresponding model response.
    """
    # Validate and parse user input using Pydantic model
    user_message = ReviewInput(review=user_input)

    # Process the user message and generate model response
    pred_score, llm_response = predict_review(user_message)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_input": user_message.review,
        "predicted_score": pred_score,
        "llm_response": llm_response
    })

def predict_review(review_data: ReviewInput) -> Tuple[int, str]:
    """
    Predicts the score of the review based on the user's input.
    :param review_data: Review input object.
    :return: Predicted review score and corresponding LLM response.
    """
    encoding = LSTM_TOOLKIT.encode_text(review_data.review)

    pred_score = MODEL(encoding["input_ids"], encoding["length"])
    pred_score = reverse_label_encoding(torch.argmax(pred_score, dim=1).detach().numpy()).item()

    return pred_score, generate_llm_response(pred_score, review_data.review)

def generate_llm_response(pred_score: str, review: str) -> str:
    """
    Generate the LLM response based on the user's review and the model's predicted score.
    :param pred_score: Predicted review score.
    :param review: User's review.
    :return: LLM response.
    """
    prompt = PROMPT_TEMPLATE.invoke({"predicted_review_score": pred_score, "review": review})
    return LLM.invoke(prompt).content


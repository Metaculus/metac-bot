#!/usr/bin/env python

import asyncio
import json
import logging
import statistics

from attr import dataclass
import requests
from decouple import config
import datetime
import re
from jinja2 import Template

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI

import argparse

# Note: To understand this code, it may be easiest to start with the `main()`
# function and read the code that is called from there.


# Dataclass for storing the metaculus API credentials + the base url of the API.
# Simplifies passing around the API info to functions.
@dataclass
class MetacApiInfo:
    token: str
    base_url: str


def build_prompt(prompt_vars: dict[str, str], news_info: str | None = None):
    """
    Function to build a prompt using a prompt template and some variables.

    Parameters:
    -----------
    prompt_vars : dict[str, str]
        a dictionary containing details about the question
    news_info: str | None, optional
        a string containing additional information gathered from external sources

    Returns:
    --------
    str
        a string containing the prompt
    """

    # the template is filled in with variables passed to `prompt_vars` and `news_info`
    PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{{title}}

background:
{{description}}

{{resolution_criteria}}

{{fine_print}}


{% if news_info %}
Your research assistant says:
{{news_info}}
{% endif %}


Today is {{today}}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) What the outcome would be if nothing changed.
(c) What you would forecast if there was only a quarter of the time left.
(d) What you would forecast if there was 4x the time left.

You write your rationale and then the last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""

    # fill template with variables using jinja2, a templating library.
    # alternatively, simple string formatting works just as well.
    prompt_jinja = Template(PROMPT_TEMPLATE)
    params = {
        "today": datetime.datetime.now().strftime("%Y-%m-%d"),
        "news_info": news_info,
        **prompt_vars,
    }
    return prompt_jinja.render(params)


def clamp(x, a, b):
    """
    Clamp a value x between a minimum value a and maximum value b.
    If x is less than a, then a is returned.
    If x is greater than b, then b is returned.
    Otherwise, x is returned.
    """
    return min(b, max(a, x))


def find_number_before_percent(s):
    """
    Convert a string like "Probability: 42%" to the number 42.
    """
    # Use a regular expression to find all numbers followed by a '%'
    matches = re.findall(r"(\d+)%", s)
    if matches:
        # Return the last number found before a '%'
        return clamp(int(matches[-1]), 1, 99)
    else:
        # Return None if no number found
        return None


def post_question_comment(api_info: MetacApiInfo, question_id: int, comment_text: str):
    """
    Post a comment on the question page as the bot user.

    Parameters:
    -----------
    api_info : MetacApiInfo
        an object containing the API info (credentials and base url)
    question_id : int
        the ID of the question to post a comment on
    comment_text : str
        the text of the comment to post

    Returns:
    --------
    json
        the JSON response from the API
    bool
        whether the request was successful
    """
    # we use the requests library to send a POST request to the comments endpoint
    url = f"{api_info.base_url}/comments/" # this is the url for the comments endpoint
    response = requests.post(
        url,
        json={
            "comment_text": comment_text,
            "submit_type": "N",
            "include_latest_prediction": True,
            "question": question_id,
        },
        headers={"Authorization": f"Token {api_info.token}"}, # your token is used to authenticate the request
    )
    if not response.ok:
        logging.error(
            f"Failed posting a comment on question {question_id}: {response.text}"
        )
    return response.json, response.ok


def post_question_prediction(
    api_info: MetacApiInfo, question_id: int, prediction_percentage: float
):
    """
    Post a prediction value on the question. The function expects a percentage
    (between 0 and 100) as input, but posts the value as a float
    (between 0 and 1).

    Parameters:
    -----------
    api_info : MetacApiInfo
        an object containing the API info (credentials and base url)
    question_id : int
        the ID of the question to post a prediction on
    prediction_percentage : float
        the prediction as a percentage (between 0 and 100) to post.

    Returns:
    --------
    json
        the JSON response from the API
    bool
        whether the request was successful
    """

    url = f"{api_info.base_url}/questions/{question_id}/predict/" # this is the url for the predict endpoint
    response = requests.post(
        url,
        json={"prediction": float(prediction_percentage) / 100},
        headers={"Authorization": f"Token {api_info.token}"},
    )
    response.raise_for_status()
    if not response.ok:
        logging.error(
            f"Failed posting a prediction on question {question_id}: {response.text}"
        )
    return response.json, response.ok


def get_question_details(api_info: MetacApiInfo, question_id):
    """
    Get all details about a specific question.
    Uses the questions endpoint, using a single question ID to return
    detailed information about the question.

    Parameters:
    -----------
    api_info : MetacApiInfo
        an object containing the API info (credentials and base url)
    question_id : int
        the ID of the question to get details on

    Returns:
    --------
    json
        the JSON response from the API containing the question details
    """
    url = f"{api_info.base_url}/questions/{question_id}/" # this is the url for the questions endpoint
    response = requests.get(
        url,
        headers={"Authorization": f"Token {api_info.token}"},
    )
    response.raise_for_status()
    return json.loads(response.content)


def list_questions(api_info: MetacApiInfo, tournament_id: int, offset=0, count=10):
    """
    List questions from a specific tournament. This uses the questions
    endpoint and queries it for questions belonging to a specific tournament.

    Parameters:
    -----------
    api_info : MetacApiInfo
        an object containing the API info (credentials and base url)
    tournament_id : int
        the ID of the tournament to list questions from
    offset : int, optional
        the number of questions to skip. This is used for pagination. I.e. if
        offset is 0 and count is 10 then the first 10 questions are returned.
        If offset is 10 and count is 10 then the next 10 questions are returned.
    count : int, optional
        the number of questions to return

    Returns:
    --------
    json
        A list of JSON objects, each containing information for a single question
    """
    # a set of parameters to pass to the questions endpoint
    url_qparams = {
        "limit": count, # the number of questions to return
        "offset": offset, # pagination offset
        "has_group": "false",
        "order_by": "-activity", # order by activity (most recent questions first)
        "forecast_type": "binary", # only binary questions are returned
        "project": tournament_id, # only questions in the specified tournament are returned
        "status": "open", # only open questions are returned
        "format": "json", # return results in json format
        "type": "forecast", # only forecast questions are returned
        "include_description": "true", # include the description in the results
    }
    url = f"{api_info.base_url}/questions/" # url for the questions endpoint
    response = requests.get(
        url,
        headers={"Authorization": f"Token {api_info.token}"},
        params=url_qparams
    )
    # you can verify what this is doing by looking at
    # https://www.metaculus.com/api2/questions/?format=json&has_group=false&limit=5&offset=0&order_by=-activity&project=3294&status=open&type=forecast
    # in the browser. The URL works as follows:
    # base_url/questions/, then a "?"" before the first url param and then a "&"
    # between additional parameters

    response.raise_for_status()
    data = json.loads(response.content)
    return data["results"]

def call_perplexity(query):
    """
    Make a call to the perplexity API to obtain additional information.

    Parameters:
    -----------
    query : str
        The query to pass to the perplexity API. This is the question we want to
        get information about.

    Returns:
    --------
    str
        The response from the perplexity API.
    """
    url = "https://api.perplexity.ai/chat/completions"
    api_key = config("PERPLEXITY_API_KEY", default="-")
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    payload = {
        "model": "llama-3.1-sonar-large-128k-chat",
        "messages": [
            {
                "role": "system", # this is a system prompt designed to guide the perplexity assistant
                "content": """
You are an assistant to a superforecaster.
The superforecaster will give you a question they intend to forecast on.
To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
You do not produce forecasts yourself.
""",
            },
            {
                "role": "user", # this is the actual prompt we ask the perplexity assistant to answer
                "content": query,
            },
        ],
    }
    response = requests.post(url=url, json=payload, headers=headers)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    print(
        f"\n\nCalled perplexity with:\n----\n{json.dumps(payload)}\n---\n, and got\n:",
        content,
    )
    return content


def get_model(model_name: str):
    """
    Get the appropriate language model based on the provided model name.
    This uses the classes provided by the llama-index library.

    Parameters:
    -----------
    model_name : str
        The name of the model to instantiate. Supported values are:
        "gpt-4o", "gpt-3.5-turbo", "anthropic", "o1-preview"

    Returns:
    --------
    Union[OpenAI, Anthropic, None]
        An instance of the specified model, or None if the model name is not recognized.

    Note:
    -----
    This function relies on environment variables for API keys. These should be
    stored in a file called ".env", which will be accessed using the
    `config` function from the decouple library.
    """

    match model_name:
        case "gpt-4o":
            return OpenAI(
                api_key=config("OPENAI_API_KEY", default=""),
                model=model_name
            )
        case "anthropic":
            tokenizer = Anthropic().tokenizer
            Settings.tokenizer = tokenizer
            return Anthropic(
                api_key=config("ANTHROPIC_API_KEY", default=""),
                model="claude-3-5-sonnet-20240620",
            )
        case "o1-preview":
            return OpenAI(
                api_key=config("OPENAI_API_KEY", default=""),
                model=model_name,
                timeout=600,
            )

    return None


async def llm_predict_once(chat_model, prompt):
    """
    Make a single prediction using the provided language model.

    Parameters:
    -----------
    chat_model : OpenAI | Anthropic | Ollama
        The language model to use for prediction. This uses the object
        returned by `get_model`.
    prompt : str
        The prompt to use for the prediction. This is the prompt generated by
        `build_prompt`.

    Returns:
    --------
    float
        The prediction as a percentage (between 0 and 100)
    str
        The rationale for the prediction
    """

    # make a call to the language model
    response = await chat_model.achat(
        messages=[ChatMessage(role=MessageRole.USER, content=prompt)]
    )

    # extract the probability from the response
    probability_match = find_number_before_percent(response.message.content)
    return probability_match, response.message.content


async def main():
    """
    Main function to run the forecasting bot. This function accesses the questions
    for a given tournament, fetches information about them, and then uses an LLM
    to generate a forecast.

    You can decide to use additional information e.g. from perplexity.ai and
    you can decide to make the final forecast the median of multiple predictions,
    instead of just a single one.

    Parameters:
    -----------
    None. All relevant parameters are passed via environment variables or
    command line arguments.

    Installing dependencies
    ----------------------
    Install poetry: https://python-poetry.org/docs/#installing-with-pipx.
    Then run `poetry install` in your terminal.

    Environment variables
    ----------------------

    You need to have a .env file with all required environment variables.

    Alternatively, if you're running the bot via github actions, you can set
    the environment variables in the repository settings.
    (Settings -> Secrets and variables -> Actions). Set API keys as secrets and
    things like the tournament id as variables.

    When using an .env file, the environment variables should be specified in the following format:
    METACULUS_TOKEN=1234567890

    The following environment variables are important:
    - METACULUS_TOKEN: your metaculus API token (go to https://www.metaculus.com/aib/ to get one)
    - TOURNAMENT_ID: the id of the tournament you want to forecast on. The Q3 id was XXXX.
    - API_BASE_URL: the base url of the metaculus API. This is https://www.metaculus.com/api2
    - OPENAI_API_KEY: your openai API key
    - ANTHROPIC_API_KEY: your anthropic API key
    - PERPLEXITY_API_KEY: your perplexity API key

    Running the bot
    ----------------------
    You can run a test version of the bot using `poetry run python forecast_bot.py`.
    By default, the bot will not submit any predictions.

    The script parses additional arguments passed to it when called. To actually
    submit predictions, run `poetry run python forecast_bot.py --submit_predictions`.

    To use perplexity, run `poetry run python forecast_bot.py --use_perplexity` etc.
    """

    # parse additional arguments passed to the script (i.e. anything after
    # `poetry run python forecast_bot.py`)
    # if you don't pass any arguments, the default values are used
    parser = argparse.ArgumentParser(
        description="A simple forecasting bot based on LLMs"
    )
    parser.add_argument(
        "--submit_predictions",
        help="Submit the predictions to Metaculus",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--use_perplexity",
        help="Use perplexity.ai to search some up to date info about the forecasted question",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--number_forecasts",
        type=int,
        default=1,
        help="The number of LLM forecasts to average per question",
    )
    parser.add_argument(
        "--metac_token_env_name",
        type=str,
        help="The name of the env variable where to read the metaculus token from",
        default="METACULUS_TOKEN",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        choices=["gpt-4o", "anthropic", "o1-preview"],
        default="gpt-4o",
        help="The model to use, one of the options listed",
    )
    parser.add_argument(
        "--metac_base_url",
        type=str,
        help="The base URL for the metaculus API",
        default=config("API_BASE_URL", default="https://www.metaculus.com/api2", cast=str),
    )
    parser.add_argument(
        "--tournament_id",
        type=int,
        help="The tournament ID where to predict",
        default=config("TOURNAMENT_ID", default=0, cast=int),
    )

    # parse the arguments and store them in the `args` variable
    args = parser.parse_args()

    # store Metaculus API info in a variable
    metac_api_info = MetacApiInfo(
        token=config(args.metac_token_env_name, default="-"),
        base_url=args.metac_base_url,
    )

    # get the language model to be used based on the arguments passed to the script (or the default)
    llm_model = get_model(args.llm_model)

    if args.number_forecasts < 1:
        print("number_forecasts must be larger than 0")
        return

    offset = 0
    # all the function logic is inside this while loop, which will run until all
    # questions in the tournament are processed
    while True:

        # get a list with information about questions in the tournament, in batches of 5.
        questions = list_questions(
            metac_api_info, args.tournament_id, offset=offset, count=5
        )
        print("Handling questions: ", [q["id"] for q in questions])
        if len(questions) < 1:
            break # break the while loop if there are no more questions to process

        offset += len(questions) # update the offset for the next batch of questions

        # for every question, we get additional information from perplexity if the
        # argument `--use_perplexity` is passed to the script.
        questions_with_news = [
            (
                question,
                call_perplexity(question["question"]["title"]) if args.use_perplexity else None,
            )
            for question in questions
        ]

        # build a prompt for every question. This prompt will be used to generate a
        # forecast for the question using an LLM.
        prompts = [
            build_prompt(
                {
                    "title": question["question"]["title"],
                    "description": question["question"]["description"],
                    "resolution_criteria": question["question"].get("resolution_criteria", ""),
                    "fine_print": question["question"].get("fine_print", ""),
                },
                news_summary,
            )
            for question, news_summary in questions_with_news
        ]

        # print the prompts to the console, so the user can verify they look correct
        for (question, _), prompt in zip(questions_with_news, prompts):
            print(
                f"\n\n*****\nPrompt for question {question['id']}/{question['question']['title']}:\n{prompt} \n\n\n\n"
            )

        # initialize a dictionary to store the predictions for each question id.
        # This looks like this: {id1: [], id2: [], id3: [], ...}
        # Later on, we will store the predictions for each question in the list
        # associated with its id.
        all_predictions = {q["id"]: [] for (q, _) in questions_with_news}

        # run the LLM `number_forecasts` times for each question and store the
        # results. By default, this is 1. But it's possible to run the LLM multiple
        # times to get a forecast that is the median of multiple runs.
        for round in range(args.number_forecasts):
            # run the LLM forecast for all questions in parallel
            results = await asyncio.gather(
                *[llm_predict_once(llm_model, prompt) for prompt in prompts],
            )
            # iterate over the llm responses and the questions
            # print the prediction and rationale to the console
            # if the prediction is not None, append it to the all_predictions list
            # submit the prediction directly if argument is set.
            # Post rationale as comment.
            for (prediction, reasoning), (question, _) in zip(results, questions_with_news):
                id = question["id"]
                print(
                    f"\n\n****\n(round {round})Forecast for {id}: {prediction}, Rationale:\n {reasoning}"
                )
                if prediction is not None:
                    all_predictions[id].append(float(prediction))
                    if args.submit_predictions:
                        post_question_prediction(metac_api_info, id, float(prediction))
                        post_question_comment(metac_api_info, id, reasoning)
                        print(f"Posted prediction for {id}")

        # if we ran the LLM more than once, compute the median of the predictions
        # across multiple runs and make the final forecast.
        if args.number_forecasts > 1:
            for question, _ in questions_with_news:
                id = question["id"]
                q_predictions = all_predictions[id]
                if len(q_predictions) < 1:
                    continue
                median = statistics.median(q_predictions)
                if args.submit_predictions:
                    # submit the final forecast and make an additional comment
                    # we submitted other forecasts already, but since this is the
                    # last one, it is the one that will be counted.
                    post_question_prediction(metac_api_info, id, median)
                    post_question_comment(
                        metac_api_info,
                        id,
                        f"Computed the median of the last {len(q_predictions)} predictions: {median}",
                    )
                    print(f"Posted final forecast for {id}")

        # iterate over all questions again and make a separate comment with the
        # perplexity info that was used to inform the forecast.
        for question, perplexity_result in questions_with_news:
            id = question["id"]
            if args.submit_predictions:
                post_question_comment(
                    metac_api_info,
                    id,
                    f"##Used perplexity info:\n\n {perplexity_result}",
                )

if __name__ == "__main__":
    asyncio.run(main())

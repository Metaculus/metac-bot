import asyncio
import datetime
import json
import os
import re
import argparse
import logging

import numpy as np
import requests
from asknews_sdk import AskNewsSDK
import typeguard
from litellm import acompletion
from litellm.files.main import ModelResponse
from litellm.types.utils import Choices
import litellm
from pydantic import BaseModel
import forecasting_tools
from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import RefreshingBucketRateLimiter

# Add this after imports, before CONSTANTS section
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

######################### CONSTANTS #########################
# Constants
SUBMIT_PREDICTION = True  # set to True to publish your predictions to Metaculus
USE_EXAMPLE_QUESTIONS = False  # set to True to forecast example questions rather than the tournament questions
NUM_RUNS_PER_QUESTION = 5  # The median forecast is taken between NUM_RUNS_PER_QUESTION runs
SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = True
GET_NEWS = True  # set to True to enable AskNews after entering ASKNEWS secrets
LLM_MODEL_NAME: str | None = None
CALL_VERY_SLOWLY = False

# Environment variables
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN") or None
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY") or None
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID") or None
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET") or None
EXA_API_KEY = os.getenv("EXA_API_KEY") or None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or None  # You'll need the OpenAI API Key if you want to use the Exa Smart Searcher


# The tournament IDs below can be used for testing your bot.
Q4_2024_AI_BENCHMARKING_ID = 32506
Q1_2025_AI_BENCHMARKING_ID = 32627
Q4_2024_QUARTERLY_CUP_ID = 3672
Q1_2025_QUARTERLY_CUP_ID = 32630
AXC_2025_TOURNAMENT_ID = 32564
GIVEWELL_ID = 3600
RESPIRATORY_OUTLOOK_ID = 3411

TOURNAMENT_ID = Q1_2025_AI_BENCHMARKING_ID

# The example questions can be used for testing your bot. (note that question and post id are not always the same)
EXAMPLE_QUESTIONS = [  # (question_id, post_id)
    (578, 578),  # Human Extinction - Binary - https://www.metaculus.com/questions/578/human-extinction-by-2100/
    (14333, 14333),  # Age of Oldest Human - Numeric - https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/
    (22427, 22427),  # Number of New Leading AI Labs - Multiple Choice - https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/
]


######################### HELPER FUNCTIONS #########################

# @title Helper functions
AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}
API_BASE_URL = "https://www.metaculus.com/api"


def post_question_comment(post_id: int, comment_text: str) -> None:
    """
    Post a comment on the question page as the bot user.
    """

    response = requests.post(
        f"{API_BASE_URL}/comments/create/",
        json={
            "text": comment_text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": post_id,
        },
        **AUTH_HEADERS,  # type: ignore
    )
    if not response.ok:
        raise RuntimeError(response.text)


def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    """
    Post a forecast on a question.
    """
    url = f"{API_BASE_URL}/questions/forecast/"
    response = requests.post(
        url,
        json=[
            {
                "question": question_id,
                **forecast_payload,
            },
        ],
        **AUTH_HEADERS,  # type: ignore
    )
    print(f"Response: {response.status_code}")
    if not response.ok:
        raise RuntimeError(response.text)


def create_forecast_payload(
    forecast: float | dict[str, float] | list[float],
    question_type: str,
) -> dict:
    """
    Accepts a forecast and generates the api payload in the correct format.

    If the question is binary, forecast must be a float.
    If the question is multiple choice, forecast must be a dictionary that
      maps question.options labels to floats.
    If the question is numeric, forecast must be a dictionary that maps
      quartiles or percentiles to datetimes, or a 201 value cdf.
    """
    if question_type == "binary":
        return {
            "probability_yes": forecast,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    if question_type == "multiple_choice":
        return {
            "probability_yes": None,
            "probability_yes_per_category": forecast,
            "continuous_cdf": None,
        }
    # numeric or date
    return {
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": forecast,
    }


def list_posts_from_tournament(
    tournament_id: int, offset: int = 0, count: int = 50
) -> list[dict]:
    """
    List (all details) {count} posts from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": ",".join(
            [
                "binary",
                "multiple_choice",
                "numeric",
            ]
        ),
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)  # type: ignore
    if not response.ok:
        raise Exception(response.text)
    data = json.loads(response.content)
    return data


def get_open_question_ids_from_tournament(tournament_id: int) -> list[tuple[int, int]]:
    posts = list_posts_from_tournament(tournament_id)

    post_dict = dict()
    for post in posts["results"]: # type: ignore
        if question := post.get("question"):
            # single question post
            post_dict[post["id"]] = [question]


    open_question_id_post_id = []  # [(question_id, post_id)]
    for post_id, questions in post_dict.items():
        for question in questions:
            if question.get("status") == "open":
                print(
                    f"ID: {question['id']}\nQ: {question['title']}\nCloses: "
                    f"{question['scheduled_close_time']}"
                )
                open_question_id_post_id.append((question["id"], post_id))

    return open_question_id_post_id


def get_post_details(post_id: int) -> dict:
    """
    Get all details about a post from the Metaculus API.
    """
    url = f"{API_BASE_URL}/posts/{post_id}/"
    print(f"Getting details for {url}")
    response = requests.get(
        url,
        **AUTH_HEADERS,  # type: ignore
    )
    if not response.ok:
        raise Exception(response.text)
    return json.loads(response.content)


llm_concurrency_semaphore: asyncio.Semaphore | None = None
rate_limiter = RefreshingBucketRateLimiter(
    capacity=1,
    refresh_rate=0.05,
)

async def call_llm(prompt: str, temperature: float = 0.3) -> str:
    assert LLM_MODEL_NAME is not None
    litellm.drop_params = True
    assert llm_concurrency_semaphore is not None

    if LLM_MODEL_NAME == "gemini/gemini-exp-1206":
        context_window = 100000
        prompt = prompt[:context_window]

    if CALL_VERY_SLOWLY:
        await rate_limiter.wait_till_able_to_acquire_resources(1)

    async with llm_concurrency_semaphore:
        response = await acompletion(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=False,
        )
        assert isinstance(response, ModelResponse)
        choices = response.choices
        choices = typeguard.check_type(choices, list[Choices])
        reasoning = choices[0].message.content
        assert isinstance(reasoning, str)
        return reasoning

def run_research(question: str) -> str:
    research = ""
    if GET_NEWS == True:
        if ASKNEWS_CLIENT_ID and ASKNEWS_SECRET:
            research = call_asknews(question)
        elif EXA_API_KEY:
            research = call_exa_smart_searcher(question)
        elif PERPLEXITY_API_KEY:
            research = call_perplexity(question)
        else:
            raise ValueError("No API key provided")
    else:
        research = "No research done"
    return research

def call_perplexity(question: str) -> str:
    url = "https://api.perplexity.ai/chat/completions"
    api_key = PERPLEXITY_API_KEY
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",  # this is a system prompt designed to guide the perplexity assistant
                "content": """
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself.
                """,
            },
            {
                "role": "user",  # this is the actual prompt we ask the perplexity assistant to answer
                "content": question,
            },
        ],
    }
    response = requests.post(url=url, json=payload, headers=headers)
    if not response.ok:
        raise Exception(response.text)
    content = response.json()["choices"][0]["message"]["content"]
    return content

def call_exa_smart_searcher(question: str) -> str:
    if OPENAI_API_KEY is None:
        searcher = forecasting_tools.ExaSearcher(
            include_highlights=True,
            num_results=10,
        )
        highlights = asyncio.run(searcher.invoke_for_highlights_in_relevance_order(question))
        prioritized_highlights = highlights[:10]
        combined_highlights = ""
        for i, highlight in enumerate(prioritized_highlights):
            combined_highlights += f'[Highlight {i+1}]:\nTitle: {highlight.source.title}\nURL: {highlight.source.url}\nText: "{highlight.highlight_text}"\n\n'
        response = combined_highlights
    else:
        searcher = forecasting_tools.SmartSearcher(
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give"
            "you a question they intend to forecast on. To be a great assistant, you generate"
            "a concise but detailed rundown of the most relevant news, including if the question"
            "would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question}"
        )
        response = asyncio.run(searcher.invoke(prompt))

    return response

def call_asknews(question: str) -> str:
    """
    Use the AskNews `news` endpoint to get news context for your query.
    The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search
    """
    ask = AskNewsSDK(
        client_id=ASKNEWS_CLIENT_ID, client_secret=ASKNEWS_SECRET, scopes=set(["news"])
    )

    # get the latest news related to the query (within the past 48 hours)
    hot_response = ask.news.search_news(
        query=question,  # your natural language query
        n_articles=6,  # control the number of articles to include in the context, originally 5
        return_type="both",
        strategy="latest news",  # enforces looking at the latest news only
    )

    # get context from the "historical" database that contains a news archive going back to 2023
    historical_response = ask.news.search_news(
        query=question,
        n_articles=10,
        return_type="both",
        strategy="news knowledge",  # looks for relevant news within the past 60 days
    )

    hot_articles = hot_response.as_dicts
    historical_articles = historical_response.as_dicts
    formatted_articles = "Here are the relevant news articles:\n\n"

    if hot_articles:
        hot_articles = [article.__dict__ for article in hot_articles]
        hot_articles = sorted(hot_articles, key=lambda x: x["pub_date"], reverse=True)

        for article in hot_articles:
            pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
            formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

    if historical_articles:
        historical_articles = [article.__dict__ for article in historical_articles]
        historical_articles = sorted(
            historical_articles, key=lambda x: x["pub_date"], reverse=True
        )

        for article in historical_articles:
            pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
            formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

    if not hot_articles and not historical_articles:
        formatted_articles += "No articles were found.\n\n"
        return formatted_articles

    return formatted_articles


class ReasonedPrediction(BaseModel):
    forecast: float | dict[str, float] | list[float]
    rationale: str

class AggregatePrediction(BaseModel):
    forecast: float | dict[str, float] | list[float]
    sub_predictions: list[ReasonedPrediction]
    news: str

############### BINARY ###############
# @title Binary prompt & functions

# This section includes functionality for binary questions.

BINARY_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

Question background:
{background}


This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
{resolution_criteria}

{fine_print}


Your research assistant says:
{summary_report}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A brief description of a scenario that results in a No outcome.
(d) A brief description of a scenario that results in a Yes outcome.

You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""


def extract_probability_from_response_as_percentage_not_decimal(
    forecast_text: str,
) -> float:
    matches = re.findall(r"(\d+)%", forecast_text)
    if matches:
        # Return the last number found before a '%'
        number = int(matches[-1])
        number = min(99, max(1, number))  # clamp the number between 1 and 99
        return number
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


async def get_binary_gpt_prediction(
    question_details: dict, num_runs: int
) -> AggregatePrediction:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]

    summary_report = run_research(title)

    content = BINARY_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
    )

    async def get_rationale_and_probability(content: str) -> ReasonedPrediction:
        rationale = await call_llm(content)
        probability = extract_probability_from_response_as_percentage_not_decimal(rationale)
        return ReasonedPrediction(
            forecast=probability/100,
            rationale=rationale
        )

    sub_predictions = await asyncio.gather(
        *[get_rationale_and_probability(content) for _ in range(num_runs)]
    )
    forecasts = [p.forecast for p in sub_predictions]
    forecasts = typeguard.check_type(forecasts, list[float])
    median_probability = float(np.median(forecasts))
    print(f"Generated {len(sub_predictions)} sub-predictions")

    return AggregatePrediction(
        forecast=median_probability,
        sub_predictions=sub_predictions,
        news=summary_report
    )


####################### NUMERIC ###############
# @title Numeric prompt & functions

NUMERIC_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

Background:
{background}

{resolution_criteria}

{fine_print}


Your research assistant says:
{summary_report}

Today is {today}.

{lower_bound_message}
{upper_bound_message}


Formatting Instructions:
- Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1m).
- Never use scientific notation.
- Always start with a smaller number (more negative if negative) and then increase from there

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The outcome if nothing changed.
(c) The outcome if the current trend continued.
(d) The expectations of experts and markets.
(e) A brief description of an unexpected scenario that results in a low outcome.
(f) A brief description of an unexpected scenario that results in a high outcome.

You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unkowns.

The last thing you write is your final answer as:
"
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"
"""

def extract_percentiles_from_response(forecast_text: str) -> dict:

    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_percentile_numbers(text) -> dict:
        pattern = r"^.*(?:P|p)ercentile.*$"
        number_pattern = r"-\s*(?:[^\d\-]*\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)|(\d+(?:,\d{3})*(?:\.\d+)?)"
        results = []

        for line in text.split("\n"):
            if re.match(pattern, line):
                numbers = re.findall(number_pattern, line)
                numbers_no_commas = [
                    next(num for num in match if num).replace(",", "")
                    for match in numbers
                ]
                numbers = [
                    float(num) if "." in num else int(num)
                    for num in numbers_no_commas
                ]
                if len(numbers) > 1:
                    first_number = numbers[0]
                    last_number = numbers[-1]
                    # Check if the original line had a negative sign before the last number
                    if "-" in line.split(":")[-1]:
                        last_number = -abs(last_number)
                    results.append((first_number, last_number))

        # Convert results to dictionary
        percentile_values = {}
        for first_num, second_num in results:
            key = first_num
            percentile_values[key] = second_num

        return percentile_values

    percentile_values = extract_percentile_numbers(forecast_text)

    if len(percentile_values) > 0:
        return percentile_values
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


def generate_continuous_cdf(
    percentile_values: dict,
    question_type: str,
    open_upper_bound: bool,
    open_lower_bound: bool,
    upper_bound: float,
    lower_bound: float,
    zero_point: float | None,
) -> list[float]:
    """
    Returns: list[float]: A list of 201 float values representing the CDF.
    """

    percentile_max = max(float(key) for key in percentile_values.keys())
    percentile_min = min(float(key) for key in percentile_values.keys())
    range_min = lower_bound
    range_max = upper_bound
    range_size = range_max - range_min
    buffer = 1 if range_size > 100 else 0.01 * range_size

    # Adjust any values that are exactly at the bounds
    for percentile, value in list(percentile_values.items()):
        if not open_lower_bound and value <= range_min + buffer:
            percentile_values[percentile] = range_min + buffer
        if not open_upper_bound and value >= range_max - buffer:
            percentile_values[percentile] = range_max - buffer

    # Set cdf values outside range
    if open_upper_bound:
        if range_max > percentile_values[percentile_max]:
            percentile_values[int(100 - (0.5 * (100 - percentile_max)))] = range_max
    else:
        percentile_values[100] = range_max

    # Set cdf values outside range
    if open_lower_bound:
        if range_min < percentile_values[percentile_min]:
            percentile_values[int(0.5 * percentile_min)] = range_min
    else:
        percentile_values[0] = range_min


    sorted_percentile_values = dict(sorted(percentile_values.items()))

    # Normalize percentile keys
    normalized_percentile_values = {}
    for key, value in sorted_percentile_values.items():
        percentile = float(key) / 100
        normalized_percentile_values[percentile] = value


    value_percentiles = {
        value: key for key, value in normalized_percentile_values.items()
    }


    # function for log scaled questions
    def generate_cdf_locations(range_min, range_max, zero_point):
        if zero_point is None:
            scale = lambda x: range_min + (range_max - range_min) * x
        else:
            deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
            scale = lambda x: range_min + (range_max - range_min) * (
                deriv_ratio**x - 1
            ) / (deriv_ratio - 1)
        xaxis = [scale(x) for x in np.linspace(0, 1, 201)]
        return xaxis

    cdf_xaxis = generate_cdf_locations(range_min, range_max, zero_point)



    def linear_interpolation(x_values, xy_pairs):
        # Sort the xy_pairs by x-values
        sorted_pairs = sorted(xy_pairs.items())

        # Extract sorted x and y values
        known_x = [pair[0] for pair in sorted_pairs]
        known_y = [pair[1] for pair in sorted_pairs]

        # Initialize the result list
        y_values = []

        for x in x_values:
            # Check if x is exactly in the known x values
            if x in known_x:
                y_values.append(known_y[known_x.index(x)])
            else:
                # Find the indices of the two nearest known x-values
                i = 0
                while i < len(known_x) and known_x[i] < x:
                    i += 1

                # If x is outside the range of known x-values, use the nearest endpoint
                if i == 0:
                    y_values.append(known_y[0])
                elif i == len(known_x):
                    y_values.append(known_y[-1])
                else:
                    # Perform linear interpolation
                    x0, x1 = known_x[i - 1], known_x[i]
                    y0, y1 = known_y[i - 1], known_y[i]

                    # Linear interpolation formula
                    y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
                    y_values.append(y)

        return y_values

    continuous_cdf = linear_interpolation(cdf_xaxis, value_percentiles)


    return continuous_cdf


async def get_numeric_gpt_prediction(
    question_details: dict, num_runs: int
) -> AggregatePrediction:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    question_type = question_details["type"]
    scaling = question_details["scaling"]
    open_upper_bound = question_details["open_upper_bound"]
    open_lower_bound = question_details["open_lower_bound"]
    upper_bound = scaling["range_max"]
    lower_bound = scaling["range_min"]
    zero_point = scaling["zero_point"]

    # Create messages about the bounds that are passed in the LLM prompt
    if open_upper_bound:
        upper_bound_message = ""
    else:
        upper_bound_message = f"The outcome can not be higher than {upper_bound}."
    if open_lower_bound:
        lower_bound_message = ""
    else:
        lower_bound_message = f"The outcome can not be lower than {lower_bound}."

    summary_report = run_research(title)

    content = NUMERIC_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        lower_bound_message=lower_bound_message,
        upper_bound_message=upper_bound_message,
    )

    async def ask_llm_to_get_cdf(content: str) -> ReasonedPrediction:
        rationale = await call_llm(content)
        percentile_values = extract_percentiles_from_response(rationale)
        cdf = generate_continuous_cdf(
            percentile_values,
            question_type,
            open_upper_bound,
            open_lower_bound,
            upper_bound,
            lower_bound,
            zero_point,
        )
        return ReasonedPrediction(
            forecast=cdf,
            rationale=rationale
        )

    sub_predictions = await asyncio.gather(
        *[ask_llm_to_get_cdf(content) for _ in range(num_runs)]
    )

    all_cdfs = np.array([p.forecast for p in sub_predictions])
    median_cdf = np.median(all_cdfs, axis=0).tolist()

    return AggregatePrediction(
        forecast=median_cdf,
        sub_predictions=sub_predictions,
        news=summary_report
    )


########################## MULTIPLE CHOICE ###############
# @title Multiple Choice prompt & functions

MULTIPLE_CHOICE_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

The options are: {options}


Background:
{background}

{resolution_criteria}

{fine_print}


Your research assistant says:
{summary_report}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A description of an scenario that results in an unexpected outcome.

You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

The last thing you write is your final probabilities for the N options in this order {options} as:
Option_A: Probability_A
Option_B: Probability_B
...
Option_N: Probability_N
"""


def extract_option_probabilities_from_response(forecast_text: str, options: list[str]) -> list[float]:

    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_option_probabilities(text: str) -> list[float]:

        # Number extraction pattern
        number_pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?"

        results = []

        # Iterate through each line in the text
        for line in text.split("\n"):
            # Extract all numbers from the line
            numbers = re.findall(number_pattern, line)
            numbers_no_commas = [num.replace(",", "") for num in numbers]
            # Convert strings to float or int
            numbers = [
                float(num) if "." in num else int(num) for num in numbers_no_commas
            ]
            # Add the tuple of numbers to results
            if len(numbers) >= 1:
                last_number = numbers[-1]
                results.append(last_number)

        return results

    option_probabilities = extract_option_probabilities(forecast_text)

    NUM_OPTIONS = len(options)

    if len(option_probabilities) > 0:
        # return the last NUM_OPTIONS items
        return option_probabilities[-NUM_OPTIONS:] # type: ignore
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


def generate_multiple_choice_forecast(options, option_probabilities) -> dict:
    """
    Returns: dict corresponding to the probabilities of each option.
    """

    # confirm that there is a probability for each option
    if len(options) != len(option_probabilities):
        raise ValueError(
            f"Number of options ({len(options)}) does not match number of probabilities ({len(option_probabilities)})"
        )

    # Ensure we are using decimals
    total_sum = sum(option_probabilities)
    decimal_list = [x / total_sum for x in option_probabilities]

    def normalize_list(float_list):
        # Step 1: Clamp values
        clamped_list = [max(min(x, 0.99), 0.01) for x in float_list]

        # Step 2: Calculate the sum of all elements
        total_sum = sum(clamped_list)

        # Step 3: Normalize the list so that all elements add up to 1
        normalized_list = [x / total_sum for x in clamped_list]

        # Step 4: Adjust for any small floating-point errors
        adjustment = 1.0 - sum(normalized_list)
        normalized_list[-1] += adjustment

        return normalized_list

    normalized_option_probabilities = normalize_list(decimal_list)

    probability_yes_per_category = {}
    for i in range(len(options)):
        probability_yes_per_category[options[i]] = normalized_option_probabilities[i]


    return probability_yes_per_category


async def get_multiple_choice_gpt_prediction(
    question_details: dict,
    num_runs: int,
) -> AggregatePrediction:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    options = question_details["options"]

    summary_report = run_research(title)

    content = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        options=options,
    )

    async def ask_llm_for_multiple_choice_probabilities(
        content: str,
    ) -> ReasonedPrediction:
        rationale = await call_llm(content)
        option_probabilities = extract_option_probabilities_from_response(rationale, options)
        probability_yes_per_category = generate_multiple_choice_forecast(
            options, option_probabilities
        )
        return ReasonedPrediction(
            forecast=probability_yes_per_category,
            rationale=rationale
        )

    sub_predictions = await asyncio.gather(
        *[ask_llm_for_multiple_choice_probabilities(content) for _ in range(num_runs)]
    )

    average_probability_yes_per_category: dict[str, float] = {}
    option_forecasts = [prediction.forecast for prediction in sub_predictions]
    option_forecasts = typeguard.check_type(option_forecasts, list[dict[str, float]])
    for option in options:
        probabilities_for_current_option = [
            forecast[option] for forecast in option_forecasts
        ]
        average_probability_yes_per_category[option] = sum(
            probabilities_for_current_option
        ) / len(probabilities_for_current_option)


    return AggregatePrediction(
        forecast=average_probability_yes_per_category,
        sub_predictions=sub_predictions,
        news=summary_report
    )


################### FORECASTING ###################
def forecast_is_already_made(post_details: dict) -> bool:
    """
    Check if a forecast has already been made by looking at my_forecasts in the question data.

    question.my_forecasts.latest.forecast_values has the following values for each question type:
    Binary: [probability for no, probability for yes]
    Numeric: [cdf value 1, cdf value 2, ..., cdf value 201]
    Multiple Choice: [probability for option 1, probability for option 2, ...]
    """
    try:
        forecast_values = post_details["question"]["my_forecasts"]["latest"][
            "forecast_values"
        ]
        return forecast_values is not None
    except Exception:
        return False

async def run_prediction_function(question_details: dict, num_runs_per_question: int) -> AggregatePrediction:
    question_type = question_details["type"]
    if question_type == "binary":
        prediction = await get_binary_gpt_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "numeric":
        prediction = await get_numeric_gpt_prediction(
            question_details, num_runs_per_question
        )
    elif question_type == "multiple_choice":
        prediction = await get_multiple_choice_gpt_prediction(
            question_details, num_runs_per_question
        )
    else:
        raise ValueError(f"Unknown question type: {question_type}")
    print(f"----------------------------------------------\n")
    print(f"Question: {question_details['title']}")
    print(f"Forecast: {prediction.forecast}")
    for sub_prediction in prediction.sub_predictions:
        print(f"Sub-Prediction: {sub_prediction.forecast}")
        print(f"Rationale: {sub_prediction.rationale}")
    print(f"News: {prediction.news}")
    return prediction

async def forecast_individual_question(
    question_id: int,
    post_id: int,
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> str:
    post_details = get_post_details(post_id)
    question_details = post_details["question"]
    title = question_details["title"]
    question_type = question_details["type"]

    summary_of_forecast = ""
    summary_of_forecast += f"----------\nQuestion: {title}\n"
    summary_of_forecast += f"URL: https://www.metaculus.com/questions/{post_id}/\n"

    if question_type == "multiple_choice":
        options = question_details["options"]
        summary_of_forecast += f"options: {options}\n"

    if (
        forecast_is_already_made(post_details)
        and skip_previously_forecasted_questions == True
    ):
        summary_of_forecast += "Skipped: Forecast already made\n"
        return summary_of_forecast

    aggregate_prediction = await run_prediction_function(question_details, num_runs_per_question)

    if submit_prediction:
        for i, sub_prediction in enumerate(aggregate_prediction.sub_predictions):
            forecast_payload = create_forecast_payload(sub_prediction.forecast, question_type)
            post_question_prediction(question_id, forecast_payload)
            comment = f"## Sub-Prediction {i+1}\n{sub_prediction.rationale}"
            post_question_comment(post_id, comment)
        forecast_payload = create_forecast_payload(aggregate_prediction.forecast, question_type)
        comment = (
            f"# Aggregate Forecast\n{str(aggregate_prediction.forecast)[:200]}\n\n"
            f"# News\n{aggregate_prediction.news}\n\n"
            f"# Individual Predictions\n"
            + "\n\n".join(f"## Sub-Prediction {i+1}\n{p.rationale}"
                         for i, p in enumerate(aggregate_prediction.sub_predictions))
        )
        post_question_prediction(question_id, forecast_payload)
        post_question_comment(post_id, comment)
        summary_of_forecast += f"Forecast: {str(aggregate_prediction.forecast)[:200]}\n"
        summary_of_forecast += "Posted: Forecast was posted to Metaculus.\n"

    return summary_of_forecast



async def forecast_questions(
    open_question_id_post_id: list[tuple[int, int]],
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> None:
    forecast_tasks = [
        forecast_individual_question(
            question_id,
            post_id,
            submit_prediction,
            num_runs_per_question,
            skip_previously_forecasted_questions,
        )
        for question_id, post_id in open_question_id_post_id
    ]
    forecast_summaries = await asyncio.gather(*forecast_tasks, return_exceptions=True)
    print("\n", "#" * 100, "\nForecast Summaries\n", "#" * 100)

    errors = []
    for question_id_post_id, forecast_summary in zip(
        open_question_id_post_id, forecast_summaries
    ):
        question_id, post_id = question_id_post_id
        if isinstance(forecast_summary, Exception):
            print(
                f"-----------------------------------------------\nPost {post_id} Question {question_id}:\nError: {forecast_summary.__class__.__name__} {forecast_summary}\nURL: https://www.metaculus.com/questions/{post_id}/\n"
            )
            errors.append(forecast_summary)
        else:
            print(forecast_summary)

    if errors:
        print("-----------------------------------------------\nErrors:\n")
        error_message = f"Errors were encountered: {errors}"
        print(error_message)
        raise RuntimeError(error_message)



######################## FINAL RUN #########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run forecasting with specified LLM model')
    parser.add_argument('--llm', type=str, help='LLM model name to use for forecasting', default=None)
    parser.add_argument('--concurrency', type=int, help='Number of concurrent LLM requests')
    parser.add_argument('--call_very_slowly', type=str, help='Make llm requests very slowly')
    parser.add_argument('--skip_previous', type=str, help='Override skip previously forecasted questions')
    parser.add_argument('--tournament_id', type=int, help='Override tournament ID', default=TOURNAMENT_ID)
    args = parser.parse_args()

    if EXA_API_KEY is not None:
        assert PERPLEXITY_API_KEY is None, "Cannot use both EXA and Perplexity"
        assert ASKNEWS_CLIENT_ID is None, "Cannot use both EXA and AskNews"
        assert ASKNEWS_SECRET is None, "Cannot use both EXA and AskNews"
        assert OPENAI_API_KEY is not None, "Need OpenAI API key for EXA"
    elif PERPLEXITY_API_KEY is not None:
        assert EXA_API_KEY is None, "Cannot use both Perplexity and EXA"
        assert ASKNEWS_CLIENT_ID is None, "Cannot use both Perplexity and AskNews"
        assert ASKNEWS_SECRET is None, "Cannot use both Perplexity and AskNews"
    elif ASKNEWS_CLIENT_ID is not None:
        assert EXA_API_KEY is None, "Cannot use both AskNews and EXA"
        assert PERPLEXITY_API_KEY is None, "Cannot use both AskNews and Perplexity"
        assert ASKNEWS_SECRET is not None, "Must provide AskNews secret"

    if args.skip_previous:
        assert args.skip_previous in ["True", "False"], "Invalid value for skip_previous. Please use 'True' or 'False'."
        SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = args.skip_previous == "True"
    if args.call_very_slowly:
        assert args.call_very_slowly in ["True", "False"], "Invalid value for call_very_slowly. Please use 'True' or 'False'."
        CALL_VERY_SLOWLY = args.call_very_slowly == "True"

    LLM_MODEL_NAME = args.llm
    llm_concurrency_semaphore = asyncio.Semaphore(args.concurrency)
    if args.tournament_id is None or args.tournament_id == 1:
        TOURNAMENT_ID = TOURNAMENT_ID # NOSONAR
    else:
        TOURNAMENT_ID = args.tournament_id

    if USE_EXAMPLE_QUESTIONS:
        open_question_id_post_id = EXAMPLE_QUESTIONS
    else:
        open_question_id_post_id = get_open_question_ids_from_tournament(TOURNAMENT_ID)

    asyncio.run(
        forecast_questions(
            open_question_id_post_id,
            SUBMIT_PREDICTION,
            NUM_RUNS_PER_QUESTION,
            SKIP_PREVIOUSLY_FORECASTED_QUESTIONS,
        )
    )


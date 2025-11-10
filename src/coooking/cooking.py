import base64
from typing import List

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Ingredient(BaseModel):
    name: str = Field(description="Name of ingredient")
    amount: str = Field(description="Amount with units")
    calories: int = Field(description="Approximate calories")


class Dish(BaseModel):
    dish_name: str = Field(description="Identified dish name")
    ingredients: List[Ingredient] = Field(description="List of ingredients")
    recipe: str = Field(description="Short textual recipe (3-5 steps)")


model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)

SYSTEM_PROMPT = """
Act as a professional chef that can recognize dishes from images.
Look at the photo and recognize the dish name. 
List the products (ingredients) needed to cook it and a short recipe with steps.
Return ONLY a structured JSON object that matches specified schema.
Do not include any explanations, only provide a RFC8259 compliant JSON response following this format without deviation.
"""

cooking_chef = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    response_format=ProviderStrategy(Dish),
)


def analyze_dish(dish_image_path: str) -> Dish:
    with open(dish_image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    message = HumanMessage(content=[
        {"type": "text", "text": "What dish is this?"},
        {
            "type": "image",
            "base64": image_base64,
            "mime_type": "image/jpeg",
        }
    ])
    return cooking_chef.invoke({"messages": [message]})


if __name__ == "__main__":
    result = analyze_dish("dishes/lazania.jpg")
    print(result["structured_response"])

import os
from openai import OpenAI
import base64
import pandas as pd

model = "gpt-4o"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

df = pd.DataFrame(
    columns=[
        "CategoryTitlePt",
        "CategoryTitleEn",
        "SubcategoryTitlePt",
        "SubcategoryTitleEn",
        "ItemNamePt",
        "ItemNameEn",
        "ItemPrice",
        "Calories",
        "PortionSize",
        "Availability",
        "ItemDescriptionPt",
        "ItemDescriptionEn",
    ]
)

# Define the system prompt
system_prompt = """
Convert the menu image to a structured excel sheet format following the provided template and instructions.
This assistant converts restaurant or cafe menu data into a structured Excel sheet that adheres to a specific template.
The template includes categories, subcategories, item names, prices, descriptions, and more, ensuring data consistency.
This assistant helps users fill out each row correctly, following the detailed instructions provided.

Overview:
- Each row in the Excel spreadsheet represents a unique item, categorized under a category or subcategory.
- Category and subcategory names are repeated for items within the same subcategory.
- Certain columns are left blank when not applicable, such as subcategory details for items directly under a category.
- Item details, including names, prices, and descriptions, must be unique for each entry.
- Uploaded menu content will be appended to the existing menu without deleting any current entries.

Columns Guide:

Column Name                    | Description                               | Accepted Values           | Example
-------------------------------|-------------------------------------------|---------------------------|-----------------------
CategoryTitlePt (Column A)      | Category names in Portuguese              | Text, 256 characters max  | Bebidas
CategoryTitleEn (Column B) (Optional) | English translations of category titles | Text, 256 characters max  | Beverages
SubcategoryTitlePt (Column C) (Optional) | Subcategory titles in Portuguese | Text, 256 characters max or blank | Sucos
SubcategoryTitleEn (Column D) (Optional) | English translations of subcategory titles | Text, 256 characters max or blank | Juices
ItemNamePt (Column E)           | Item names in Portuguese                  | Text, 256 characters max  | Água Mineral
ItemNameEn (Column F) (Optional) | English translations of item names | Text, 256 characters max or blank | Mineral Water
ItemPrice (Column G)          | Price of each item without currency symbol  | Text                      | 2.50 or 2,50
Calories (Column H) (Optional) | Caloric content of each item              | Numeric                   | 150
PortionSize (Column I)        | Portion size for each item in units        | Text                      | 500ml, 1, 2-3
Availability (Column J) (Optional) | Current availability of the item     | Numeric: 1 for Yes, 0 for No | 1
ItemDescriptionPt (Column K) (Optional) | Detailed description in Portuguese | Text, 500 characters max  | Contains essential minerals
ItemDescriptionEn (Column L) (Optional) | Detailed description in English | Text, 500 characters max  | Contains essential minerals

Notes:
- Ensure all data entered follows the specified formats to maintain database integrity.
- Review the data for accuracy and consistency before submitting the Excel sheet.
"""

# Get the directory named "Dim Sum"
directory = os.path.join(os.path.dirname(__file__), "Dim Sum")

os.chdir(directory)
IMAGE_DIR = directory


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Process each image in the directory
image_files = sorted(
    [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
)

test = image_files[0]

import os
import openai
import pydantic


class DescriptionOutput(pydantic.BaseModel):
    one_word: str
    two_words: str
    three_words: str
    four_words: str
    five_words: str
    is_common_household_object: bool


class ChatGPTDescriber:
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18", api_key=None) -> None:
        """
        model:
            Cheap model:
                gpt-4o-mini-2024-07-18
            Expensive model:
                gpt-4o-2024-11-20
        api_key:
            OpenAI API key. If None, it will try to use the OPENAI_API_KEY environment variable.
        """
        self.set_api_key(api_key)

        self.model = model

    def set_api_key(self, api_key):
        try:
            self.client = openai.OpenAI(api_key=api_key)
        except openai.OpenAIError as e:
            print(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable to use ChatGPTDescriber or call ChatGPTDescriber.set_api_key()."
            )

    def describe(self, descriptions) -> DescriptionOutput:
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Using the specified amount of words describe the object in simple words keeping the essentials of the visual appearance (e.g. the main color) in the context of an object laying on a table. Use at least one noun. Also mark whether it's a common household object or not.",
                },
                {
                    "role": "user",
                    "content": "Description of the object: " + " ".join(descriptions),
                },
            ],
            response_format=DescriptionOutput,
        )
        gpt_output = completion.choices[0].message.parsed  # type: DescriptionOutput
        return gpt_output


chatgpt_describer = ChatGPTDescriber()

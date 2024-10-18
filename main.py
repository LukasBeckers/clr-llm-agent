from agents.BaseClassifier import BaseClassifier

from agents.LLMs import gpt_4o_mini


if __name__=="__main__":
    sentiment_classifier = BaseClassifier(
        valid_classes=["Neutral", "Positive", "Negative"],
        llm = gpt_4o_mini
    )

    text = "Ich finde diese Arbeit ok!"

    result = sentiment_classifier(text)
    print(result, type(result))

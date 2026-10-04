from core.llm.router import get_llm


def test_llm(provider):

    print(f"\nTesting {provider}...")

    llm = get_llm(provider)

    response = llm.invoke(
        "Explain machine learning in one sentence."
    )

    print(response.content)


if __name__ == "__main__":

    test_llm("openai")
    test_llm("gemini")
    test_llm("claude")
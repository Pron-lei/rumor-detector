from src.explain import generate_explanation


samples = [
    (
        "BREAKING: Unconfirmed reports say a second explosion happened downtown.",
        1,
        0.86,
    ),
    (
        "The city council will hold a public meeting on Friday according to the official schedule.",
        0,
        0.82,
    ),
]


for text, label, confidence in samples:
    print(generate_explanation(text, label, confidence, use_llm=False))
    print()

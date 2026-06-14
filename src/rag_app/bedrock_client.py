import boto3


def build_prompt(question: str, matched_documents: list[str]) -> str:
    context = "\n\n---\n\n".join(matched_documents)

    return f"""
Use the context below to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""


def ask_bedrock(
    question: str,
    results,
    model_id: str,
    region: str,
    max_tokens: int = 700,
    temperature: float = 0.2,
) -> str:
    matched_documents = results["documents"][0]
    prompt = build_prompt(question, matched_documents)

    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name=region,
    )

    response = bedrock.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    )

    return response["output"]["message"]["content"][0]["text"]

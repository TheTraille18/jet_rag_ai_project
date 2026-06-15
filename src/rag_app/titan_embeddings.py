import json

import boto3


def get_bedrock_runtime(region: str):
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=region,
    )


def embed_text(
    text: str,
    bedrock_runtime,
    model_id: str,
    dimensions: int = 1024,
    normalize: bool = True,
) -> list[float]:
    body = {
        "inputText": text,
        "dimensions": dimensions,
        "normalize": normalize,
    }

    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response["body"].read())
    return response_body["embedding"]


def embed_texts(
    texts: list[str],
    bedrock_runtime,
    model_id: str,
    dimensions: int = 1024,
    normalize: bool = True,
) -> list[list[float]]:
    return [
        embed_text(
            text=text,
            bedrock_runtime=bedrock_runtime,
            model_id=model_id,
            dimensions=dimensions,
            normalize=normalize,
        )
        for text in texts
    ]

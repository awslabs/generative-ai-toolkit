import warnings

import boto3
import numpy as np
import pytest
from lib.agent.agent import TheCityAgent

from generative_ai_toolkit.test.mock import MockBedrockConverse
from generative_ai_toolkit.tracer import InMemoryTracer

warnings.filterwarnings("ignore", category=DeprecationWarning)


class MockEmbeddingModel:
    embedding_dimension: int
    mocked_embedding: np.ndarray | None

    def __init__(self, embedding_dimension: int):
        self.embedding_dimension = embedding_dimension
        self.mocked_embedding = None

    def encode(self, texts: list[str]):
        if self.mocked_embedding is not None:
            return np.repeat(self.mocked_embedding, len(texts), axis=0)
        return np.random.rand(len(texts), self.embedding_dimension).astype(np.float32)

    def start_mocking(self):
        self.mocked_embedding = np.random.rand(1, self.embedding_dimension)

    def stop_mocking(self):
        self.mocked_embedding = None


@pytest.fixture(scope="session")
def mock_bedrock_converse():
    """
    Initiates mock of agent.
    """
    mock = MockBedrockConverse()
    yield mock
    if mock.trajectory:
        raise Exception("Unconsumed mock trajectory")


@pytest.fixture(scope="session")
def mock_embedding_model():
    """
    Initiates mock of embedding model.
    """
    model = MockEmbeddingModel(512)
    yield model


@pytest.fixture(scope="module")
def city_agent(mock_bedrock_converse, mock_embedding_model):
    """
    Initiate real agent
    """
    agent = TheCityAgent(
        model_id="eu.anthropic.claude-3-sonnet-20240229-v1:0",
        session=mock_bedrock_converse.session(),
        load_documents=True,
        embedding_model=mock_embedding_model,
        embedding_dimension=mock_embedding_model.embedding_dimension,
    )
    yield agent


@pytest.fixture(scope="module")
def city_agent_zwei(mock_bedrock_converse, mock_embedding_model):
    """
    Initiate real agent
    """
    agent = TheCityAgent(
        model_id="eu.anthropic.claude-3-sonnet-20240229-v1:0",
        session=mock_bedrock_converse.session(),
        embedding_model=mock_embedding_model,
        embedding_dimension=mock_embedding_model.embedding_dimension,
    )
    yield agent


@pytest.fixture(scope="module")
def unmocked_city_agent(mock_embedding_model):
    """
    Initiate an unmocked agent
    """
    agent = TheCityAgent(
        model_id="eu.anthropic.claude-3-sonnet-20240229-v1:0",
        session=boto3.session.Session(region_name="eu-central-1"),
        load_documents=True,
        embedding_model=mock_embedding_model,
        embedding_dimension=mock_embedding_model.embedding_dimension,
        tracer=InMemoryTracer,
        default_model=True,
    )
    yield agent

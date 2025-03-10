from dataclasses import dataclass, field
from typing import Any, Tuple

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

@dataclass
class Schema:
    labels: list[str]


@dataclass
class Entity:
    token_start_idx: int
    token_end_idx: int
    label: str
    text: str

@dataclass
class RawDocument:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    entities: set[str] = field(default_factory=list)
    relations: set[Tuple[str, str, str]] = field(default_factory=list)


NER_instruction = """
You are an expert annotator. Your task is to extract **entity labels** from unstructured text.
- Identify meaningful entity types (e.g., person, organization, event, location).
- Include implicit entities (e.g., roles, quantities) based on context.
- Return the labels as a **Python list** with single quotes around each label, enclosed in square brackets, and separated by commas.
- Example output: `['person', 'organization', 'event']`.
- Ensure no duplicates and no additional text outside the list.
- Return labels in language that is equal to the input text language.
"""



ER_instruction = """
You are an expert annotator. Your task is to extract **relation labels** between entities in the text.
- Use the provided entities and context to infer meaningful relationships.
- Return the labels as a **Python list** with single quotes around each label, enclosed in square brackets, and separated by commas.
- Example output: `['founded_by', 'located_in', 'acquired']`.
- Ensure no duplicates and no additional text outside the list.
"""

NEW_ER_instruction = """
You are an expert annotator. Your task is to extract **relation labels** between entities in the text.
- Use the provided entities and context to infer meaningful relationships.
- Example output: `labels = {"glirel_labels": {
    "co-founder": {"allowed_head": ["person"], "allowed_tail": ["ORG"]},
    "no relation": {},  # head and tail can be any entity type
    "country of origin": {"allowed_head": ["person", "ORG"], "allowed_tail": ["LOC", "GPE"]},
    "parent": {"allowed_head": ["person"], "allowed_tail": ["person"]},
    "located in or next to body of water": {"allowed_head": ["LOC", "GPE", "FAC"], "allowed_tail": ["LOC", "GPE"]},
    "spouse": {"allowed_head": ["person"], "allowed_tail": ["person"]},
    "child": {"allowed_head": ["person"], "allowed_tail": ["person"]},
    "founder": {"allowed_head": ["person"], "allowed_tail": ["ORG"]},
    "founded on date": {"allowed_head": ["ORG"], "allowed_tail": ["DATE"]},
    "headquartered in": {"allowed_head": ["ORG"], "allowed_tail": ["LOC", "GPE", "FAC"]},
    "acquired by": {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "person"]},
    "subsidiary of": {"allowed_head": ["ORG"], "allowed_tail": ["ORG", "person"]}
    }
}`.
- Ensure no duplicates and no additional text outside the list.
"""


# --------------------- NER Examples ---------------------
EXAMPLE_TEXT2 = """
At the International Innovation Conference in San Francisco on 10 August 2019, renowned entrepreneur Elon Musk, CEO of SpaceX, and Bill Gates, co-founder of Microsoft, engaged in a high-stakes discussion on breakthroughs in artificial intelligence and renewable energy. In a separate keynote, Google announced its acquisition of YouTube in 2006, marking a pivotal moment in digital media history.
"""
EXAMPLE_NER_OUTPUT2 = """['event', 'location', 'date', 'person', 'role', 'organization', 'company', 'acquisition', 'year']"""

EXAMPLE_TEXT3 = """
On March 15, 2022, BioGen Inc., a leading pharmaceutical company based in Boston, announced the FDA approval of NeuroCure, a groundbreaking treatment for Alzheimer's disease. The clinical trials, conducted in partnership with Harvard University, showed a 50% reduction in symptoms.
"""
EXAMPLE_NER_OUTPUT3 = """['date', 'organization', 'location', 'drug', 'disease', 'university', 'percentage']"""

EXAMPLE_TEXT4 = """
During the 2023 CES in Las Vegas, Tesla unveiled their new SolarDrone technology, which integrates AI-powered navigation systems developed in collaboration with NVIDIA.
"""
EXAMPLE_NER_OUTPUT4 = """['event', 'year', 'location', 'company', 'product', 'technology']"""


# --------------------- ER Examples ---------------------
EXAMPLE_DOCUMENT1 = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""
EXAMPLE_RELATIONS1 = """['located_in', 'is_a', 'established_on', 'broadcasts_in', 'expanded_to', 'initiated', 'provides', 'forayed_into', 'forayed_in']"""
EXAMPLE_ENTITIES1 = """['company', 'country', 'date',, 'language', 'website']"""
EXAMPLE_NEW_RELATIONS1 = """{
    "glirel_labels": {
        "located_in": {"allowed_head": ["company", "website"], "allowed_tail": ["country", "region"]},
        "is_a": {"allowed_head": ["company", "website"], "allowed_tail": ["company", "radio station"]},
        "established_on": {"allowed_head": ["company"], "allowed_tail": ["date", "time"]},
        "broadcasts_in": {"allowed_head": ["company"], "allowed_tail": ["language"]},
        "expanded_to": {"allowed_head": ["company"], "allowed_tail": ["website", "digital media"]},
        "initiated": {"allowed_head": ["company"], "allowed_tail": ["website", "project"]},
        "provides": {"allowed_head": ["website"], "allowed_tail": ["service", "content"]},
        "forayed_into": {"allowed_head": ["company"], "allowed_tail": ["digital media", "new media"]},
        "forayed_in": {"allowed_head": ["company"], "allowed_tail": ["date", "time"]}
    }
}"""


EXAMPLE_DOCUMENT2 = """
In 2020, EcoTech Solutions, a renewable energy startup founded by Clara Rodriguez, secured a $20 million investment from Green Ventures. This partnership aims to accelerate the development of solar-powered desalination plants in drought-affected regions like Cape Town.
"""
EXAMPLE_ENTITIES2 = """['EcoTech Solutions', 'renewable energy startup', 'Clara Rodriguez', '2020', '$20 million', 'Green Ventures', 'solar-powered desalination plants', 'drought-affected regions', 'Cape Town']"""
EXAMPLE_RELATIONS2 = """Example output ['founded_by', 'invested_by', 'partnered_with', 'develops', 'located_in', 'dated_in']"""
EXAMPLE_NEW_RELATIONS2 = """{
    "glirel_labels": {
        'founder': {"allowed_head": ["PERSON"], "allowed_tail": ["ORG"]},
        "founded_on_date': {"allowed_head": ["ORG"], "allowed_tail": ["DATE"]},
        "acquired by": {"allowed_head": ["ORG"], "allowed_tail": ["ORG"]},
        "headquartered_in": {"allowed_head": ["ORG"], "allowed_tail": ["GPE"]}
    }
}"""

EXAMPLE_DOCUMENT3 = """
The collaboration between NASA and SpaceX, initiated under the Artemis program in 2021, has significantly advanced lunar exploration technologies, including the Orion spacecraft and Starship lunar lander.
"""
EXAMPLE_RELATIONS3 = """['collaborates_with', 'initiated_under', 'dated_in', 'enhanced', 'comprises', 'initiated_in']"""
EXAMPLE_ENTITIES3 = """['company', 'programme', 'date', 'technology', 'spacecraft']"""
EXAMPLE_NEW_RELATIONS3 = """{
    "glirel_labels": {
        "collaborates_with": {"allowed_head": ["company"], "allowed_tail": ["company"]},
        "initiated_under": {"allowed_head": ["company"], "allowed_tail": ["programme"]},
        "dated_in": {"allowed_head": ["programme"], "allowed_tail": ["date"]},
        "enhanced": {"allowed_head": ["company"], "allowed_tail": ["technology"]},
        "comprises": {"allowed_head": ["technology"], "allowed_tail": ["spacecraft"]},
        "initiated_in": {"allowed_head": ["company"], "allowed_tail": ["date"]}
    }
}"""

EXAMPLE_DOCUMENT4 = """
Following lengthy negotiations, BankCorp finalized its merger with FinTech Global in Q3 2023, creating the largest digital banking entity in Europe. The deal, valued at €4.5 billion, was brokered by CEO Maria Schmidt and FinTech's founder, Alexei Petrov.
"""
EXAMPLE_ENTITIES4 = """['company', 'date', 'country', 'money', 'person', 'role']"""
EXAMPLE_RELATIONS4 = """['merged_with', 'created', 'valued_at', 'negotiated_by', 'has_role', 'located_in', 'merged_in']"""
EXAMPLE_NEW_RELATIONS4 = """{
    "glirel_labels": {
        "merged_with": {"allowed_head": ["company"], "allowed_tail": ["company"]},
        "create": {"allowed_head": ["company"], "allowed_tail": ["company"]},
        "valued_at": {"allowed_head": ["company"], "allowed_tail": ["money"]},
        "negotiated_by": {"allowed_head": ["company"], "allowed_tail": ["person"]},
        "has_role": {"allowed_head": ["person"], "allowed_tail": ["role"]},
        "located_in": {"allowed_head": ["company"], "allowed_tail": ["country"]},
        "merged_in": {"allowed_head": ["company"], "allowed_tail": ["date"]}
    }
}
"""

# --------------------- Prompt Templates ---------------------
ER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(ER_instruction),
    HumanMessage(f"{EXAMPLE_DOCUMENT1}\n\nEntities: {EXAMPLE_ENTITIES1}"),
    AIMessage(EXAMPLE_RELATIONS1),
    HumanMessage(f"{EXAMPLE_DOCUMENT3}\n\nEntities: {EXAMPLE_ENTITIES3}"),
    AIMessage(EXAMPLE_RELATIONS3),
    HumanMessage(f"{EXAMPLE_DOCUMENT4}\n\nEntities: {EXAMPLE_ENTITIES4}"),
    AIMessage(EXAMPLE_RELATIONS4),
    HumanMessagePromptTemplate.from_template("{user_input}\n\nEntities: {entities}")
])

ER_NEW_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(NEW_ER_instruction),
    HumanMessage(f"{EXAMPLE_DOCUMENT1}\n\nEntities: {EXAMPLE_ENTITIES1}"),
    AIMessage(EXAMPLE_NEW_RELATIONS1),
    HumanMessage(f"{EXAMPLE_DOCUMENT3}\n\nEntities: {EXAMPLE_ENTITIES3}"),
    AIMessage(EXAMPLE_NEW_RELATIONS3),
    HumanMessage(f"{EXAMPLE_DOCUMENT4}\n\nEntities: {EXAMPLE_ENTITIES4}"),
    AIMessage(EXAMPLE_NEW_RELATIONS4),
    HumanMessagePromptTemplate.from_template("{user_input}\n\nEntities: {entities}")
])

NER_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(NER_instruction),
    HumanMessage(EXAMPLE_TEXT2),
    AIMessage(EXAMPLE_NER_OUTPUT2),
    HumanMessage(EXAMPLE_TEXT3),
    AIMessage(EXAMPLE_NER_OUTPUT3),
    HumanMessage(EXAMPLE_TEXT4),
    AIMessage(EXAMPLE_NER_OUTPUT4),
    HumanMessagePromptTemplate.from_template("{user_input}")
])

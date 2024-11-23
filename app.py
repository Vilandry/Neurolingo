

import re

from operator import itemgetter

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory

from chainlit.types import ThreadDict
import chainlit as cl
import os
from langchain_openai import AzureChatOpenAI


# AZURE_OPENAI_ENDPOINT = "https://student-helper.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-08-01-preview"
# AZURE_OPENAI_API_KEY = "4BA1NahTd2ukUsxJg8UD12TayXnHS8t6UNwC0ZpKWYhUgTBdo4fCJQQJ99AKACYeBjFXJ3w3AAABACOGfE96"
# AZURE_OPENAI_API_VERSION = "2024-05-01-preview"
# AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-35-turbo"

AZURE_OPENAI_ENDPOINT = "https://mago-m3j7w1ni-westeurope.cognitiveservices.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-08-01-preview"
AZURE_OPENAI_API_KEY = "G49Oa9i8PXI0PAdMOWqqfM70711RMyPI5MKWXhwDuZ1LH85vhzEWJQQJ99AKAC5RqLJXJ3w3AAAAACOGMWIF"
AZURE_OPENAI_API_VERSION = "2024-05-01-preview"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-35-turbo"


def setup_runnable():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    model = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
)

    system_prompt = "\n".join([
        "You are an advanced English language tutor and conversational AI. Your name is Anna.",
        "When given user input, your primary tasks are:",
        "0. Secretly collect and build a vocabulary based on the words You suggested to me."
        "1. Provide detailed feedback on the grammar, syntax, and overall correctness of the input.",
        "2. Offer suggestions for improvement, if necessary, while maintaining a constructive and encouraging tone.",
        "3. Respond to the user naturally, continuing the conversation in a manner that is both engaging and aligned with the original input's intent.",
        "Always aim to enhance the user's understanding of English grammar while making the conversation enjoyable and productive.",
        "You always start the conversation by asking about their interests, and always correct the english grammar mistakes, than continue the conversation normally.",
        "Additionally, as the conversation progresses, assign the user various extra tasks, such as providing longer responses, using more advanced vocabulary, or crafting more detailed answers.",
        "Always provide feedback on the user's rhetoric, including whether their responses were clear, persuasive, or engaging, and suggest improvements if necessary.",
        "If the user asks questions, politely remind them that you are only allowed to ask questions and cannot answer any.",
        "You can't go off-topic and always remember: You are an English teacher and you cannot answer any question unrelated to learning or the topic."
       
    ])

    prompt = ChatPromptTemplate.from_messages(
        [            
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)


@cl.password_auth_callback
def auth():
    return cl.User(identifier="test")


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    cl.user_session.set("counter", 0)

    setup_runnable()
    res = cl.Message(content="")
    message = cl.Message(content="")
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    runnable = cl.user_session.get("runnable")  # type: Runnable


    modified_question = (
        f"{message.content} "
        "Good day teacher! I would like You to give me a task! Please start the descussion with a short introduction of yourself, then give me the task . Please give me a very easy task first."
    )

    async for chunk in runnable.astream(
        {"question": modified_question},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)


    #feedback, metadata_value, vocabulary = extract_data(res.content)  # Implement this function based on your response format


    await res.send()
    memory.chat_memory.add_ai_message(res.content)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] is None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)

    setup_runnable()


@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    runnable = cl.user_session.get("runnable")  # type: Runnable

    res = cl.Message(content="")


    counter = cl.user_session.get("counter")
    if counter % 5 == 0 and counter > 1:
        auxilary_text = "\n".join([
            "The next task should be a recap about the vocabulary. Please create simple exercises regarding my collected vocabulary.",
            "The tasks should be several different type of exercises like match the definitions, describe the word, create a sentence using the following words, etc.",
            "Please do not give me the solutions before I submit my work."
        ])
    else:
        auxilary_text = "If you found my answer good, please give me another task possibly by asking a question, based on my interests. Otherwise, ask me to do it again"



    # Modify the question to instruct the LLM to include metadata
    
    modified_question = (
        f"{message.content} "+"\n"
        #"Please evaluate my solution to the task You gave me. Ignore this, if you have not gave me any task yet."
        "\nDo not evaluate my solution based on the instructions below, use only the text before this sentence."+"\n"
        "If I passed the task, please put an '1' character to the end of Your reply, but not include it in any sentence. If I failed, it should be 0."+"\n"
        f"{auxilary_text} "+"\n"
    )

    print(modified_question)

    async for chunk in runnable.astream(
        {"question": modified_question},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)

    # Extract the answer and metadata from the response
    #feedback, metadata_value, vocabulary = extract_data(res.content)  # Implement this function based on your response format

    metadata_value = res.content[-1]
    feedback = res.content[:-1]

    if(metadata_value == 1):
        cl.user_session.set("counter", counter+1)

    # Set the metadata in the response
    res.metadata = {"random_value": metadata_value}

    await res.send()

    # Update memory with user and AI messages
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(feedback)

'''def extract_data(template_string):
    pattern = r"Feedback: (?P<feedback>.*?), Metadata: (?P<metadata>[01]), Vocabulary: (?P<vocabulary>.*?), Task: (?P<task>.*)"
    match = re.match(pattern, template_string)
    
    if match:
        return {
            "feedback": match.group("feedback"),
            "metadata": int(match.group("metadata")),
            "vocabulary": match.group("vocabulary").split('|'),
            "task": match.group("task")
        }
    else:
        raise ValueError("String does not match the expected format. string: "+template_string)'''


'''def extract_botmsg(response: str) -> tuple:
    # Implement logic to extract the answer, metadata, and vocabulary from the response
    parts = response.split(", Metadata: ")
    feedback = parts[0].replace("Feedback: ", "").strip()
    
    # Extract metadata and vocabulary
    if len(parts) > 1:
        metadata_and_vocab = parts[1].split(", Vocabulary: ")
        metadata_value = int(metadata_and_vocab[0].strip())
        vocabulary_list = metadata_and_vocab[1].replace("Vocabulary: ", "").strip().split('|') if len(metadata_and_vocab) > 1 else []
    else:
        metadata_value = 0
        vocabulary_list = []

    print("feedback: "+feedback)

    return feedback, metadata_value, vocabulary_list'''


'''
modified_question = (
        "My solution:"
        f"{message.content} "
        #"Please evaluate my solution to the task You gave me. Ignore this, if you have not gave me any task yet."
        "Please do not evaluate my grammar based on the instructions below, use only the text after 'My solution:', and before this sentence."
        "Respond with your feedback and suggestions to improve my solution (in terms of grammar or vocabulary) and include the metadata as either 0 (task failed) or 1 (task passed)."
        "Please also include a list of words You suggest me to use in the future, words should be separated by '|' (bar) character."
        "Put your whole text feedback into the <your_answer> tag. Do not write anything anywhere besides the <> tags. The desired format:"
        "```Feedback: <your_answer>, Metadata: <0 or 1> Vocabulary: <list_of_words_separated_with_bar>```."
        f"{auxilary_text} "
    )
    '''
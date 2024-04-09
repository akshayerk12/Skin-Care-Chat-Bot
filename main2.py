import os
from constants import openai_key, google_api,new_google_api
from langchain.llms import GooglePalm
# from langchain.llms import openai
import streamlit as st
# from langchain_community.llms import CTransformers
# from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field



class Tips(BaseModel):
    answer: str=Field(description='curing methods')



parser=JsonOutputParser(pydantic_object=Tips)

API_KEY=new_google_api  #os.environ['google_api']
st.title('Demo')
input_text=st.text_input('Search Topic you want')
# BARD_API_KEY='fAg8XkT_F-9p-brCYwm1gvdoNOx0uUidxvsEF_k4ROxHzfazdQ0poP1O7LvXBV4VgxXboQ'

llm=GooglePalm(
    google_api_key=API_KEY,
    temperature=0.5
)
# llm=CTransformers(model='llama-2-7b-chat.ggmlv3.q8_0.bin',
#                   model_type='llama',
#                   config={'max_new_tokens':256,'temperature':0.01})
# a=llm('Who is Messi')
# print(a)
# person_memory=ConversationBufferMemory(input_key='condition',memory_key='chat_history')
# dob_memory=ConversationBufferMemory(input_key='morning_tips',memory_key='chat_history')
# descr_memory=ConversationBufferMemory(input_key='afternoon_tips',memory_key='description_history')



firstprompt=PromptTemplate(
    input_variables=['condition'],
    template='You need to act as a Doctor. user will give you a skin condition and suggest a detailed routine for cure that condition. \n{format_instructions}\n{condition}: This is my current skin condition help me.',
    partial_variables={"format_instructions": parser.get_format_instructions()},

)
chain=firstprompt | llm | parser

# secondpromt=PromptTemplate(
#     input_variables=['morning_tips'],
#     template='Based on the {morning_tips} give me 3 tips for afternoon to get rid. Note: try to not include same tips from {morning_tips}',
# )

# thirdpromt=PromptTemplate(
#     input_variables=['afternoon_tips'],
#     template='Based on {afternoon_tips} give me 3 more tips for night.'
# )

# chain=LLMChain(llm=llm,
#          prompt=firstprompt,
#          verbose=True,
#          output_key='morning_tips')

# chain2=LLMChain(llm=llm,
#                 prompt=secondpromt,
#                 verbose=True,
#                 output_key='afternoon_tips')

# chain3=LLMChain(llm=llm,
#                 prompt=thirdpromt,
#                 verbose=True,
#                 output_key='evening_tips')

# parent_chain=SequentialChain(input_variables=['condition'],chains=[chain,chain2],verbose=True,output_variables=['morning_tips','afternoon_tips'])


if input_text:
    # st.write(parent_chain({'condition':input_text}))
    st.write(chain.invoke({'condition':input_text}))
    # with st.expander('Person Name'):
    #     st.info(person_memory.buffer)
    # with st.expander('Major Events'):
    #     st.info(descr_memory.buffer)
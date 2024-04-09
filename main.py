import os
from constants import openai_key, google_api,new_google_api
from langchain.llms import GooglePalm
# from langchain.llms import openai
import streamlit as st
from langchain_community.llms import CTransformers
# from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory

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
person_memory=ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory=ConversationBufferMemory(input_key='person',memory_key='chat_history')
descr_memory=ConversationBufferMemory(input_key='yearofbirth',memory_key='description_history')



firstprompt=PromptTemplate(
    input_variables=['name'],
    template='Tell me about the celebrity {name}'
)



chain=LLMChain(llm=llm,
         prompt=firstprompt,
         verbose=True,
         output_key='person',memory=person_memory)

secondpromt=PromptTemplate(
    input_variables=['person'],
    template='When was {person} born'
)

thirdpromt=PromptTemplate(
    input_variables=['yearofbirth'],
    template='Mention 5 important events happend around {yearofbirth} in the world'
)

chain2=LLMChain(llm=llm,
                prompt=secondpromt,
                verbose=True,
                output_key='yearofbirth',memory=dob_memory)

chain3=LLMChain(llm=llm,
                prompt=thirdpromt,
                verbose=True,
                output_key='description',memory=descr_memory)

parent_chain=SequentialChain(input_variables=['name'],chains=[chain],verbose=True,output_variables=['person'])


# llm=OpenAI(temperature=0.8)
if input_text:
    st.write(parent_chain({'name':input_text}))
    with st.expander('Person Name'):
        st.info(person_memory.buffer)
    with st.expander('Major Events'):
        st.info(descr_memory.buffer)
import os
from constants import openai_key, google_api,new_google_api
from langchain.llms import GooglePalm
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import json

API_KEY=new_google_api 
st.title('Your Skin Doctor')
input_text=st.text_input('Talk about your condition and feelings')

llm=GooglePalm(
    google_api_key=API_KEY,
    temperature=0.5
)

class Tips(BaseModel):
    answer: str=Field(description='curing methods')



parser=JsonOutputParser(pydantic_object=Tips)

firstprompt=PromptTemplate(
    input_variables=['condition'],
    template='''Act as a Doctor. user has given a skin condition: {condition}.Recommend a skin care routine and best food items (with correct names) to be included.
    If the condition is not related to skin or health say I dont know.
    Convert the output to the format \n{format_instructions}''',
    partial_variables={"format_instructions": parser.get_format_instructions()},

)
chain=firstprompt | llm | parser
file_path='results.json'
if input_text:
    # st.write(parent_chain({'condition':input_text}))
    answer=chain.invoke({'condition':input_text})
    st.write(answer)
    with open(file_path,'w') as json_file:
        json.dump(answer,json_file)

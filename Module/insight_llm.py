#!/usr/bin/env python
# coding: utf-8

# In[5]:


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


def llm(system_prompt,prompt):
    
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2

    )

    messages = [
        (
            "system",system_prompt    
        ),
        ("human", prompt,)
    ]
    ai_msg = llm.invoke(messages)
    
    return ai_msg.content


# In[ ]:





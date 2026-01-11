import json
import numpy as np
import pandas as pd
from typing import List
from openai import OpenAI
import os
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
with open('embs_subset.json') as f:
    embs=json.load(f)
embs_arr=np.array(embs)
df=pd.read_csv('df_subset.csv')
df.assign(**{'description':df['description'].apply(lambda val:val[:500])})

from fastapi import FastAPI
app=FastAPI()

@app.post("/top")
async def top(item_names:List[str],k:int=10):
    embs_query=np.array(embed(item_names)).T
    scores=embs_arr@embs_query
    top_indices=np.argpartition(scores, -k,axis=0)[-k:]
    prompts=[map_results_to_resolution_prompt(top_indices[:,i].tolist(),item_names[i]) for i in range(len(item_names))]
    responses=[client.chat.completions.create(model='gpt-4o-mini',messages=[{'role':'system','content':prompt}]).choices[0].message.content
               for prompt in prompts
               ]
    postprocessed=[try_parse_int(z) for z in responses]
    # return responses,postprocessed,prompts
    indices_in_df=[int(top_indices[z,i]) if z!=None else None for i,z in enumerate(postprocessed)]
    matched_df_rows=[df.iloc[z] if z!=None else None for z in indices_in_df]
    return matched_df_rows

def top_test(item_names:List[str],k:int=10):
    embs_query=np.array(embed(item_names)).T
    scores=embs_arr@embs_query
    top_indices=np.argpartition(scores, -k,axis=0)[-k:]
    prompts=[map_results_to_resolution_prompt(top_indices[:,i].tolist(),item_names[i]) for i in range(len(item_names))]
    responses=[client.chat.completions.create(model='gpt-4o-mini',messages=[{'role':'system','content':prompt}]).choices[0].message.content
               for prompt in prompts
               ]
    postprocessed=[try_parse_int(z) for z in responses]
    # return responses,postprocessed,prompts
    indices_in_df=[int(top_indices[z,i]) if z!=None else None for i,z in enumerate(postprocessed)]
    matched_df_rows=[df.iloc[z] if z!=None else None for z in indices_in_df]
    return matched_df_rows

def embed(lst:List[str]):
    response=client.embeddings.create(
        model='text-embedding-3-small',
        input=lst,
    )
    return [z.embedding for z in response.data]

def try_parse_int(s:str)->int|None:
    try:
        return int(s)
    except:
        return None

def map_results_to_resolution_prompt(row_of_ixs:List[int],item_name:str):
    prompt=f"""We are a manufacturing automation business. We extracted that the customer asked for "{item_name}". We cosine similarity matched it to the following top items:

    {df.iloc[row_of_ixs].reset_index(drop=True).reset_index().to_json(orient='records',indent=4)}

    If one of them are what the user asked for, output its index as an int, 0-9. Otherwise, output the string "NONE". E.g., the user could have said "two-and-a-half inch fire lock T", and that would match "2 1/2 FIRELOCK TEE", so if it had index=5, you would output 5. Please don't output an index unless there is a strong semantic match. Other examples: query "three-quarter inch chrome up cut chin" would match "3/4 Chrome Cup 401 Escutcheon" because they sound the same (transcription isn't perfect), query "half-inch gate valve whole part" would match "1/2 BRZ GATE VLV TE FULL PRT". One bug that you run into is matching user query "b" to part name "2 1\\/2 FIRELOCK TEE", which doesn't make sense, don't do that.
    """
    return prompt


# print(top_test(['7060 for partitioned grooved T','a','b','ten wafer butterfly valve with switch']))
# print(top_test(['7012 eight partitioned gasket flange','Two-by-two-by-one 6000 first third T']))
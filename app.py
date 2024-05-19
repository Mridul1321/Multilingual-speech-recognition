import streamlit as st
import moviepy.editor as me
import librosa
import numpy as np
import gc

#whisper 
import torch 
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

#llm model
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

#embeddings
from nltk.tokenize import sent_tokenize
import re
import pandas as pd
from sentence_transformers import util, SentenceTransformer
from tqdm import tqdm



if 'image_processed' not in st.session_state:
    st.session_state.image_processed=False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_input' not in st.session_state:
    st.session_state.user_input=''
if 'embedded' not in st.session_state:
    st.session_state.embedded=False

if 'llm_model' not in st.session_state:
    st.session_state.llm_model=None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer=None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text=None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model=None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings=None
if 'chunks' not in st.session_state:
    st.session_state.chunks=None
if 'model_whisper' not in st.session_state:
    st.session_state.model_whisper=None
if 'query_input' not in st.session_state:
    st.session_state.query_input=None

def clear_gpu_memoty(model_whisper):
    del st.session_state.model_whisper
    gc.collect()
    torch.cuda.empty_cache()
    st.session_state.model_whisper=None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []


#start


def get_file():
      #getting the file 
    def auduio_file(file):
        if uploaded_file_name[-4:] in ['.mp4','.mkv','.mov'] :
            
            video=me.VideoFileClip(uploaded_file_name)
            audio=video.audio
            audio.write_audiofile('temp.mp3')
        file_path = 'temp.mp3'
        audio_data, sampling_rate = librosa.load(file_path, sr=16000)  # sr=None to preserve original sampling rate

        audio_info = {
            'path': file_path,
            'array': audio_data,
            'sampling_rate': sampling_rate
        }
        return audio_info

    uploaded_file=st.file_uploader("Upload a audio or Video file",type=['mp3','wav','mp4','mkv','mov'])

    if uploaded_file is not None:
        uploaded_file_name = uploaded_file.name
        uploaded_file_name="temp"+uploaded_file_name[-4:]
        with open(uploaded_file_name, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        audio_file=auduio_file(uploaded_file_name)
        # st.write(audio_file)

        get_text(audio_file)

#getting the text form thr audio file
def get_text(audio_path):
      
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32

    model_id_whisper = "openai/whisper-large-v3"

    model_whisper = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id_whisper, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model_whisper.to(device)

    processor_whisper = AutoProcessor.from_pretrained(model_id_whisper)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_whisper,
        tokenizer=processor_whisper.tokenizer,
        feature_extractor=processor_whisper.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    result=pipe(audio_path,generate_kwargs={"task": "translate"})
    #return  result["text"]


    st.session_state.model_whisper=model_whisper


    # st.write(f'Text Extracted forom the File \n{result["text"]}')
   
   
    print('Text Extraceted')
    st.session_state.extracted_text=result['text']
    # print(text)
    st.session_state['messages'].append({"role": "bot", "content": f'The content extracted form the audio file :\n\n{result["text"]}'})

    clear_gpu_memoty(model_whisper)
    summarizer(result["text"])











def mode_creaction():
  
    model_id = "google/gemma-2b-it"
    attn_implementation = "sdpa"
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, 
                                                    torch_dtype=torch.float16, # datatype to use, we want float16
                                                    quantization_config=quantization_config ,
                                                    low_cpu_mem_usage=False, # use full memory 
                                                    attn_implementation=attn_implementation)
    

    # print("Model Created ")
    st.session_state.llm_model=llm_model
    st.session_state.tokenizer=tokenizer







#LLM moodel starts
def prompt_formatter_summ(text):
  
    base_prompt = """Based on the following text, please give the summary of the text.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
\nNow use the following text to summarize the text:
{text}
Answer:"""
    base_prompt = base_prompt.format(text=text)
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]
    tokenizer=st.session_state.tokenizer
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt

def summarizer(text,temperature=0.7,
        max_new_tokens=512,
        format_answer_text=True, 
        return_answer_only=True):
    mode_creaction()
    prompt = prompt_formatter_summ(text)
    # print(prompt)
    tokenizer=st.session_state.tokenizer
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")   
    # print(input_ids)
    llm_model=st.session_state.llm_model
    outputs = llm_model.generate(**input_ids,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens)
    output_text = tokenizer.decode(outputs[0])
    output_text_nex = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here's the summary you requested:\n\n", "")
    # return output_text

    # st.write(f'The summery of the file is :\n{output_text_nex}')
    st.session_state['messages'].append({"role": "bot", "content": f'The Summery of the file :\n{output_text_nex}'})
    st.session_state.image_processed=True
    print('text return sucess')

    embeddeing()


def embeddeing():
    text=st.session_state.extracted_text
    sentences_of_text=sent_tokenize(text)
    num_sentence_chunk_size = 10 
    def split_list(input_list ,slice_size) :
        return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]
    sentence_chunks = split_list(sentences_of_text,num_sentence_chunk_size)
    num_chunks = len(sentence_chunks)
    chunks = []
    for sentence_chunk in sentence_chunks:
        chunk_dict = {}
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) 
        chunk_dict["sentence_chunk"] = joined_sentence_chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 
        chunks.append(chunk_dict)
    df = pd.DataFrame(chunks)
    df.describe().round(2)
    min_token_length = 30
    chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", 
                                      device="cpu")
    embedding_model.to("cuda")
    for i in tqdm(chunks_over_min_token_len):
        i["embedding"] = embedding_model.encode(i["sentence_chunk"])
    text_chunks_and_embeddings_df = pd.DataFrame(chunks_over_min_token_len)
    embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
    text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_chunks_and_embedding_df = pd.read_csv("text_chunks_and_embeddings_df.csv")

    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

    chunks = text_chunks_and_embedding_df.to_dict(orient="records")
    embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)

    print('Text embedded success')
    st.session_state.embedding_model=embedding_model
    st.session_state.embeddings=embeddings
    st.session_state.chunks=chunks


def prompt_formatter_rag(query,context_items):
  
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:
"""
    base_prompt = base_prompt.format(context=context, query=query)
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]
    tokenizer=st.session_state.tokenizer
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt

def retriever_score(query):
    embedding_model=st.session_state.embedding_model
    embeddings=st.session_state.embeddings
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]
    print(len(embeddings))
    if len(embeddings)<4:
        k=1
    else :
        k=4
    scores, indices = torch.topk(input=dot_scores, k=k)
    return scores,indices


def rag_answers( query,
        temperature=0.7,
        max_new_tokens=512,
        format_answer_text=True):
    
    chunks=st.session_state.chunks
    tokenizer=st.session_state.tokenizer
    llm_model=st.session_state.llm_model
    scores, indices = retriever_score(query)
    context_items = [chunks[i] for i in indices]
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu()
    prompt = prompt_formatter_rag(query,context_items)
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")   
    outputs = llm_model.generate(**input_ids,
                                 temperature=temperature,
                                 do_sample=True,
                                 max_new_tokens=max_new_tokens)
    output_text = tokenizer.decode(outputs[0])
    if format_answer_text:
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here are the relevant passages from the context:\n\n", "")
    # print(output_text)
    return output_text




def main():
      
    st.title("TensorGo Assidment")
    if not st.session_state.image_processed:
        get_file()
    
    st.write("Chat with the file")

    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("You:", "")
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        st.session_state['messages'].append({"role": "user", "content": user_input})
        if user_input.lower()=='exit':
            st.session_state.image_processed=None
            st.session_state['messages']=[]
        response = rag_answers(user_input)
        st.session_state['messages'].append({"role": "bot", "content": response})

    for message in st.session_state['messages']:
        if message["role"] == "user":
            st.write(f"You: {message['content']}")
        else:
            st.write(f"Bot: {message['content']}")
        
    
   
            
if __name__=='__main__':
    main()
    
    
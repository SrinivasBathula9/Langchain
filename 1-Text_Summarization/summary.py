from pytube import YouTube
import time
import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader

st.set_page_config(
    layout="wide"
)
# Set the title and background color
st.title("LangChain: Summarize Text From YT or Website 🎥")
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
st.subheader('Built with LangChain🦜and Groq ❤️')
st.markdown('<style>h3{color: pink;  text-align: center;}</style>', unsafe_allow_html=True)

## Get the Groq API Key and url(Youtube or website)to be summarized
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")

# Expander for app details
with st.expander("About the App"):
    st.write("This app allows you to summarize while watching a YouTube video or Website .")
    st.write("Enter a YouTube URL in the input box below and click 'Submit' to start. This app is built by AI Anytime.")


# Input box for YouTube URL
generic_url = st.text_input("URL",label_visibility="collapsed")

## Gemma Model USsing Groq API
llm =ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template="""
Provide a summary of the following content in 1000 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    start_time = time.time() 
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()

                ## Chain For Summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)#stuff
                output_summary=chain.run(docs)

                #st.success(output_summary)
                end_time = time.time()  # End the timer
                elapsed_time = end_time - start_time
                # Display layout with 2 columns
                col1, col2 = st.columns([1,1])

                # Column 1: Video view
                with col1:
                   st.video(generic_url) # Display the video


                # Column 2: Summary View
                with col2:
                    st.header("Summarization of YouTube Video or Website")
                    #st.write(output_summary)
                    st.success(output_summary)
                    st.write(f"Time taken: {elapsed_time:.2f} seconds")
        except Exception as e:
            st.exception(f"Exception:{e}")

    
# 🌍 Multilingual Speech Recognition with RAG (Without Training)

This project implements a **Multilingual Speech Recognition system integrated with Retrieval-Augmented Generation (RAG)** without training any speech model from scratch. It leverages **OpenAI Whisper v3** for multilingual speech-to-text, performs **automatic translation to English**, and enables **summarization and question answering** using an LLM through RAG.

The application is built as a **Streamlit web app**, capable of handling **audio and video inputs** across multiple languages.

---

## 🚀 Features

* 🎙️ Multilingual speech recognition (audio & video)
* 🌐 Supports multiple languages: **English, Hindi, Tamil, Telugu, etc.**
* 🔄 Automatic translation of all inputs to English
* 🧠 Vector embedding and Retrieval-Augmented Generation (RAG)
* 📝 Speech summarization using **Gemma-2B-IT** LLM
* ❓ Question answering from uploaded speech content
* 🖥️ Interactive **Streamlit UI**
* ⚡ No training required – fully pre-trained models

---

## 🧠 System Architecture

1. Audio / Video Input
2. Multilingual Speech-to-Text using **Whisper v3**
3. Translation to English
4. Text Embedding into Vector Store
5. RAG-based Retrieval
6. LLM (Gemma-2B-IT) for:

   * Summarization
   * Question Answering

---

## 📂 Project Structure

```
Multilingual-speech-recognition/
├── app.py                                # Streamlit application
├── start.ipynb                           # Experimental / testing notebook
├── requirements.txt                     # Python dependencies
├── test_video/                          # Sample audio/video files
├── Multilingual Speech Recognition.pdf  # Project documentation
└── README.md
```

---

## 🛠️ System Requirements

* **RAM:** 8 GB or higher
* **Processor:** Intel Core i5 (12th Gen or higher)
* **GPU:** NVIDIA RTX 3050 or higher
* **VRAM:** Minimum 8 GB
* **OS:** Windows
* **CUDA:** Properly configured (mandatory)

---

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/Mridul1321/Multilingual-speech-recognition.git
cd Multilingual-speech-recognition
```

2. **Set up CUDA**

* Ensure NVIDIA GPU drivers are installed
* CUDA and cuDNN must be properly configured

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🎯 How to Use

Follow the steps below to run the Streamlit application:

1. Clone the project from GitHub:

```
https://github.com/Mridul1321/Multilingual-speech-recognition
```

2. Ensure **CUDA is installed and working**, and your system has:

* NVIDIA GPU
* VRAM greater than **8 GB**

3. Install all dependencies in the CUDA-enabled environment:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

5. The application will start running on **localhost** and open automatically in your browser.

6. Upload an **audio or video file** in any supported language.

7. The application will:

* Convert speech to text
* Translate it to English
* Generate a **summary** of the content

8. Enter your **questions** related to the uploaded speech.

9. The system performs **RAG**, retrieves relevant information, and generates accurate answers using the LLM.

---

## 📌 Demo Video

A sample demo video of the application is available here:

🔗 **Google Drive Demo:**
[https://drive.google.com/drive/folders/1-PPN__0hmG-BbU55w9wg2htvXt-T9lIK](https://drive.google.com/drive/folders/1-PPN__0hmG-BbU55w9wg2htvXt-T9lIK)

---

## 📖 Methodology

* **Whisper v3** is used for multilingual speech recognition.
* All recognized speech is translated into **English**.
* Text is converted into **vector embeddings**.
* **Retrieval-Augmented Generation (RAG)** retrieves relevant chunks.
* **Gemma-2B-IT LLM** generates:

  * Speech summaries
  * Context-aware answers to user queries

---

## ✅ Conclusion

This project successfully demonstrates a **multilingual speech recognition system integrated with RAG**, capable of handling audio and video inputs without any model training. By combining **Whisper**, **vector embeddings**, and **LLMs**, the system delivers accurate transcription, translation, summarization, and intelligent question answering through an easy-to-use Streamlit interface.

---

⭐ *If you find this project useful, consider giving it a star on GitHub!*

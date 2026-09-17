import streamlit as st
import pandas as pd
import altair as alt

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


@st.cache_data
def load_lexicon():
    """Memuat lexicon InSet (word, weight) dari lexicon_id.csv menjadi dictionary."""
    df = pd.read_csv("lexicon_id.csv")
    return dict(zip(df["word"].str.lower(), df["weight"]))


@st.cache_resource
def get_sastrawi_tools():
    stemmer = StemmerFactory().create_stemmer()
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, stopword_remover


def preprocess_id(text):
    """Lowercase -> hapus stopword -> stemming ke kata dasar."""
    stemmer, stopword_remover = get_sastrawi_tools()
    text = text.lower()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text


def analyze_sentiment_id(raw_text, lexicon):
    """Lexicon-based sentiment scoring untuk teks Bahasa Indonesia."""
    processed = preprocess_id(raw_text)
    tokens = processed.split()

    score = 0
    pos_list = []
    neg_list = []
    neu_list = []
    for word in tokens:
        weight = lexicon.get(word)
        if weight is None:
            neu_list.append(word)
        elif weight > 0:
            score += weight
            pos_list.append((word, weight))
        elif weight < 0:
            score += weight
            neg_list.append((word, weight))
        else:
            neu_list.append(word)

    polarity = score / len(tokens) if tokens else 0
    subjectivity = (len(pos_list) + len(neg_list)) / len(tokens) if tokens else 0

    return {
        "processed_text": processed,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "positives": pos_list,
        "negatives": neg_list,
        "neutral": neu_list,
    }


def convert_to_df(polarity, subjectivity):
    sentiment_dict = {"polarity": polarity, "subjectivity": subjectivity}
    return pd.DataFrame(sentiment_dict.items(), columns=["metric", "value"])


def main():
    st.title("Sentiment Analysis NLP App")
    st.subheader("Analisis Sentimen Bahasa Indonesia (InSet Lexicon + Sastrawi)")

    lexicon = load_lexicon()

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        with st.form("nlpFormID"):
            raw_text = st.text_area("Masukkan Teks Di Sini")
            submit_button = st.form_submit_button(label='Analisis')

        col1, col2 = st.columns(2)
        if submit_button and raw_text.strip():
            result = analyze_sentiment_id(raw_text, lexicon)

            with col1:
                st.info("Hasil")
                st.write(f"Teks setelah preprocessing: *{result['processed_text']}*")
                st.write({"polarity": result["polarity"], "subjectivity": result["subjectivity"]})

                if result["polarity"] > 0:
                    st.markdown("Sentimen:: Positif :smile: ")
                elif result["polarity"] < 0:
                    st.markdown("Sentimen:: Negatif :angry: ")
                else:
                    st.markdown("Sentimen:: Netral :neutral_face: ")

                result_df = convert_to_df(result["polarity"], result["subjectivity"])
                st.dataframe(result_df)

                c = alt.Chart(result_df).mark_bar().encode(
                    x='metric', y='value', color='metric')
                st.altair_chart(c, use_container_width=True)

            with col2:
                st.info("Token Sentiment")
                st.write({
                    "positives": result["positives"],
                    "negatives": result["negatives"],
                    "neutral": result["neutral"],
                })
    else:
        st.subheader("About")
        st.write(
            "Aplikasi ini menganalisis sentimen teks Bahasa Indonesia menggunakan "
            "InSet Lexicon (Koto & Rahmaningtyas, 2017) yang berisi ribuan kata "
            "positif/negatif dengan bobot -5 s.d. +5, dikombinasikan dengan "
            "Sastrawi untuk stemming dan penghapusan stopword sebelum pencocokan "
            "ke lexicon."
        )


if __name__ == '__main__':
    main()
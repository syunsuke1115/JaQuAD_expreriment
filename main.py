import streamlit as st
import torch
import warnings
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#インターフェース
def main():
    st.header("JaQuad")
    st.text("質問を入れてください")
    
    question = st.text_input('質問')
    context = st.text_input('問題文')
    model = AutoModelForQuestionAnswering.from_pretrained('SkelterLabsInc/bert-base-japanese-jaquad')
    tokenizer = AutoTokenizer.from_pretrained('SkelterLabsInc/bert-base-japanese-jaquad')
    
    if st.button("Submit"):
        if question is None or context is None:
            st.error('質問と問題文を入力してください')
            return
        inputs = tokenizer(question, context,add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        
        st.subheader('質問')
        st.write(question)
        st.subheader('問題文')
        st.write(context)
        st.subheader('回答')
        st.write(answer)
        
if __name__ == '__main__':
    main()
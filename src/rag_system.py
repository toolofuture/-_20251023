"""
병원 고객 질의응답 RAG 시스템
"""
import os
import pickle
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.schema import Document
import yaml
import streamlit as st

class HospitalRAGSystem:
    """병원 RAG 시스템 클래스"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """RAG 시스템 초기화"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        
    def setup_embeddings(self):
        """임베딩 모델 설정"""
        try:
            # OpenAI 임베딩 사용
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            print("✅ OpenAI 임베딩 모델 설정 완료")
        except Exception as e:
            print(f"❌ 임베딩 모델 설정 실패: {e}")
            # 대체 임베딩 모델 사용
            from sentence_transformers import SentenceTransformer
            self.embeddings = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ SentenceTransformer 임베딩 모델 설정 완료")
    
    def load_qa_data(self, data_path: str) -> List[Dict]:
        """Q&A 데이터 로드"""
        try:
            with open(data_path, 'rb') as f:
                qa_pairs = pickle.load(f)
            print(f"✅ Q&A 데이터 로드 완료: {len(qa_pairs)}개 샘플")
            return qa_pairs
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return []
    
    def create_documents(self, qa_pairs: List[Dict]) -> List[Document]:
        """Document 객체 생성"""
        documents = []
        
        for qa in qa_pairs:
            # 질문과 답변을 결합한 텍스트 생성
            content = f"질문: {qa['question']}\n답변: {qa['answer']}"
            
            doc = Document(
                page_content=content,
                metadata={
                    'id': qa['id'],
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'category': qa['category'],
                    **qa['metadata']
                }
            )
            documents.append(doc)
        
        return documents
    
    def setup_vectorstore(self, documents: List[Document]):
        """벡터 저장소 설정"""
        try:
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['model']['chunk_size'],
                chunk_overlap=self.config['model']['chunk_overlap']
            )
            
            splits = text_splitter.split_documents(documents)
            print(f"✅ 문서 분할 완료: {len(splits)}개 청크")
            
            # 벡터 저장소 생성
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.config['data']['vectorstore_path']
            )
            
            # 검색기 설정
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config['rag']['top_k']}
            )
            
            print("✅ 벡터 저장소 설정 완료")
            
        except Exception as e:
            print(f"❌ 벡터 저장소 설정 실패: {e}")
    
    def setup_qa_chain(self):
        """Q&A 체인 설정"""
        try:
            llm = OpenAI(
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                model_name=self.config['model']['llm_model'],
                temperature=0.1,
                max_tokens=self.config['rag']['max_tokens']
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True
            )
            
            print("✅ Q&A 체인 설정 완료")
            
        except Exception as e:
            print(f"❌ Q&A 체인 설정 실패: {e}")
    
    def query(self, question: str) -> Dict[str, Any]:
        """질의응답 처리"""
        try:
            if self.qa_chain is None:
                return {
                    'answer': 'RAG 시스템이 초기화되지 않았습니다.',
                    'source_documents': [],
                    'confidence': 0.0
                }
            
            # 질의 처리
            result = self.qa_chain({"query": question})
            
            # 결과 정리
            answer = result['result']
            source_docs = result['source_documents']
            
            # 신뢰도 계산 (간단한 휴리스틱)
            confidence = min(1.0, len(source_docs) / self.config['rag']['top_k'])
            
            return {
                'answer': answer,
                'source_documents': source_docs,
                'confidence': confidence,
                'question': question
            }
            
        except Exception as e:
            return {
                'answer': f'질의 처리 중 오류가 발생했습니다: {str(e)}',
                'source_documents': [],
                'confidence': 0.0,
                'question': question
            }
    
    def initialize_system(self):
        """전체 시스템 초기화"""
        print("🚀 병원 RAG 시스템 초기화 시작...")
        
        # 1. 임베딩 설정
        self.setup_embeddings()
        
        # 2. 데이터 로드
        train_qa_pairs = self.load_qa_data("data/processed/train_qa_pairs.pkl")
        
        if not train_qa_pairs:
            print("❌ 훈련 데이터를 찾을 수 없습니다.")
            return False
        
        # 3. 문서 생성
        documents = self.create_documents(train_qa_pairs)
        
        # 4. 벡터 저장소 설정
        self.setup_vectorstore(documents)
        
        # 5. Q&A 체인 설정
        self.setup_qa_chain()
        
        print("✅ 병원 RAG 시스템 초기화 완료!")
        return True

# Streamlit용 RAG 시스템 인스턴스
@st.cache_resource
def get_rag_system():
    """Streamlit에서 사용할 RAG 시스템 인스턴스"""
    rag_system = HospitalRAGSystem()
    rag_system.initialize_system()
    return rag_system

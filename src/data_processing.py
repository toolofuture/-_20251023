"""
병원 고객 질의응답 데이터 전처리 모듈
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
import yaml
import os

class HospitalDataProcessor:
    """병원 데이터 전처리 클래스"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """설정 파일 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def load_data(self, train_path: str, validation_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터 로드"""
        try:
            train_data = pd.read_csv(train_path, encoding='utf-8')
            val_data = pd.read_csv(validation_path, encoding='utf-8')
            
            print(f"✅ 훈련 데이터 로드 완료: {len(train_data)}개 샘플")
            print(f"✅ 검증 데이터 로드 완료: {len(val_data)}개 샘플")
            
            return train_data, val_data
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return None, None
    
    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if pd.isna(text):
            return ""
        
        # 기본 정리
        text = str(text).strip()
        
        # 특수문자 정리 (의료 용어 보존)
        text = re.sub(r'[^\w\s가-힣.,!?]', '', text)
        
        # 연속 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def create_qa_pairs(self, df: pd.DataFrame) -> List[Dict]:
        """Q&A 쌍 생성"""
        qa_pairs = []
        
        for idx, row in df.iterrows():
            question = self.preprocess_text(row.get('질문', ''))
            answer = self.preprocess_text(row.get('답변', ''))
            
            if question and answer:
                qa_pairs.append({
                    'id': idx,
                    'question': question,
                    'answer': answer,
                    'category': 'hospital',
                    'metadata': {
                        'source': 'hospital_qa',
                        'index': idx
                    }
                })
        
        return qa_pairs
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """데이터 품질 분석"""
        analysis = {
            'total_samples': len(df),
            'missing_questions': df['질문'].isna().sum(),
            'missing_answers': df['답변'].isna().sum(),
            'avg_question_length': df['질문'].str.len().mean(),
            'avg_answer_length': df['답변'].str.len().mean(),
            'unique_questions': df['질문'].nunique(),
            'duplicate_questions': df['질문'].duplicated().sum()
        }
        
        return analysis
    
    def save_processed_data(self, qa_pairs: List[Dict], output_path: str):
        """전처리된 데이터 저장"""
        import pickle
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(qa_pairs, f)
        
        print(f"✅ 전처리된 데이터 저장 완료: {output_path}")

def main():
    """메인 실행 함수"""
    processor = HospitalDataProcessor()
    
    # 데이터 로드
    train_data, val_data = processor.load_data(
        "data/raw/병원_train.csv",
        "data/raw/병원_validation.csv"
    )
    
    if train_data is not None:
        # 데이터 품질 분석
        train_analysis = processor.analyze_data_quality(train_data)
        print("📊 훈련 데이터 품질 분석:")
        for key, value in train_analysis.items():
            print(f"  {key}: {value}")
        
        # Q&A 쌍 생성
        train_qa_pairs = processor.create_qa_pairs(train_data)
        val_qa_pairs = processor.create_qa_pairs(val_data)
        
        # 전처리된 데이터 저장
        processor.save_processed_data(train_qa_pairs, "data/processed/train_qa_pairs.pkl")
        processor.save_processed_data(val_qa_pairs, "data/processed/val_qa_pairs.pkl")
        
        print(f"✅ 전처리 완료: {len(train_qa_pairs)}개 훈련 샘플, {len(val_qa_pairs)}개 검증 샘플")

if __name__ == "__main__":
    main()

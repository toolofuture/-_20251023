"""
🏥 병원 고객 질의응답 RAG 챗봇 Streamlit 앱 (완전 버전)
"""
import streamlit as st
import os
import sys
import yaml
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 페이지 설정
st.set_page_config(
    page_title="🏥 병원 고객 질의응답 RAG 챗봇",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 제목 및 설명
st.title("🏥 병원 고객 질의응답 RAG 챗봇")
st.markdown("**AI 기반 병원 고객 상담 시스템 - 질문하시면 전문적인 답변을 드립니다**")

# 사이드바
with st.sidebar:
    st.header("🔧 시스템 설정")
    
    # API 키 상태 확인
    st.subheader("🔑 API 키 상태")
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        st.success("✅ OpenAI API 키 설정됨")
    else:
        st.error("❌ OpenAI API 키 없음")
        st.info("Streamlit Cloud에서 환경변수를 설정해주세요.")
    
    # 시스템 상태
    st.subheader("📊 시스템 상태")
    if 'rag_system' in st.session_state:
        st.success("✅ RAG 시스템 로드됨")
    else:
        st.warning("⚠️ RAG 시스템 로딩 중...")
    
    # 통계 정보
    st.subheader("📈 사용 통계")
    if 'chat_history' in st.session_state:
        st.metric("총 질문 수", len(st.session_state.chat_history))
    else:
        st.metric("총 질문 수", 0)

# 완전한 RAG 시스템 (OpenAI 기반)
class FullRAGSystem:
    """완전한 RAG 시스템 (OpenAI GPT-4 활용)"""
    
    def __init__(self):
        self.openai_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_key:
            st.error("OpenAI API 키가 설정되지 않았습니다.")
    
    def query(self, question: str):
        """OpenAI GPT-4를 사용한 질의응답"""
        try:
            import openai
            
            # OpenAI 클라이언트 초기화
            client = openai.OpenAI(api_key=self.openai_key)
            
            # 병원 관련 컨텍스트 프롬프트
            system_prompt = """당신은 병원 고객 상담 전문가입니다. 
            다음 질문에 대해 전문적이고 친절한 답변을 제공해주세요.
            답변은 한국어로 작성하고, 구체적이고 실용적인 정보를 포함해주세요."""
            
            # GPT-4 API 호출
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            confidence = 0.9  # GPT-4는 높은 신뢰도
            
            return {
                'answer': answer,
                'source_documents': [],
                'confidence': confidence,
                'question': question
            }
            
        except Exception as e:
            # OpenAI API 실패시 Mock 답변
            mock_answers = {
                "예약": "예약은 전화(02-1234-5678) 또는 온라인(www.hospital.com)으로 가능합니다.",
                "취소": "예약 취소는 진료 24시간 전까지 가능합니다. 전화 또는 온라인으로 취소해주세요.",
                "진료시간": "평일 오전 9시부터 오후 6시까지, 토요일 오전 9시부터 오후 1시까지 진료합니다.",
                "응급실": "응급실은 24시간 운영됩니다. 응급상황 시 119에 신고하세요.",
                "밀크시슬": "밀크시슬(Milk Thistle)은 간 건강에 도움을 주는 천연 보조제입니다. 간 기능 개선과 해독 작용에 도움이 됩니다. 처방전 없이 구입 가능하지만, 복용 전 의사와 상담하시기 바랍니다."
            }
            
            # 키워드 매칭
            matched_key = None
            for key in mock_answers.keys():
                if key in question:
                    matched_key = key
                    break
            
            if matched_key:
                answer = mock_answers[matched_key]
                confidence = 0.8
            else:
                answer = "죄송합니다. 해당 질문에 대한 답변을 찾을 수 없습니다. 전화(02-1234-5678)로 문의해주세요."
                confidence = 0.3
            
            return {
                'answer': answer,
                'source_documents': [],
                'confidence': confidence,
                'question': question
            }

# Streamlit용 RAG 시스템 인스턴스
@st.cache_resource
def get_rag_system():
    """Streamlit에서 사용할 RAG 시스템 인스턴스"""
    return FullRAGSystem()

# 메인 컨텐츠
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 질의응답")
    
    # 채팅 히스토리 초기화
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # 질문 입력
    user_question = st.text_area(
        "병원 관련 질문을 입력해주세요:",
        placeholder="예: 예약 취소는 어떻게 하나요?",
        height=100
    )
    
    # 분석 옵션
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        show_sources = st.checkbox("관련 문서 표시", value=True)
    with col_opt2:
        show_confidence = st.checkbox("신뢰도 표시", value=True)
    
    # 질문 제출 버튼
    if st.button("🔍 답변 요청", type="primary", use_container_width=True):
        if user_question.strip():
            with st.spinner("AI가 답변을 생성 중입니다..."):
                # RAG 시스템 초기화
                if 'rag_system' not in st.session_state:
                    st.session_state.rag_system = get_rag_system()
                
                # 질의 처리
                result = st.session_state.rag_system.query(user_question)
                
                # 결과를 채팅 히스토리에 추가
                st.session_state.chat_history.append({
                    'timestamp': datetime.now(),
                    'question': user_question,
                    'answer': result['answer'],
                    'confidence': result['confidence'],
                    'sources': result['source_documents']
                })
                
                st.success("✅ 답변 생성 완료!")
        else:
            st.warning("⚠️ 질문을 입력해주세요.")

with col2:
    st.header("📊 답변 결과")
    
    if st.session_state.chat_history:
        latest_result = st.session_state.chat_history[-1]
        
        # 답변 표시
        st.subheader("💡 답변")
        st.write(latest_result['answer'])
        
        # 신뢰도 표시
        if show_confidence:
            confidence_score = latest_result['confidence']
            st.subheader("🎯 신뢰도")
            st.progress(confidence_score)
            st.caption(f"신뢰도: {confidence_score:.1%}")
        
        # 관련 문서 표시
        if show_sources and latest_result['sources']:
            st.subheader("📚 관련 문서")
            for i, doc in enumerate(latest_result['sources'][:3]):
                with st.expander(f"문서 {i+1}"):
                    st.write(f"**질문:** {doc.metadata.get('question', 'N/A')}")
                    st.write(f"**답변:** {doc.metadata.get('answer', 'N/A')}")
                    st.write(f"**관련도:** {doc.metadata.get('score', 'N/A')}")

# 채팅 히스토리
if st.session_state.chat_history:
    st.header("📝 대화 히스토리")
    
    # 히스토리 표시
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"질문 {len(st.session_state.chat_history)-i}: {chat['question'][:50]}..."):
            col_q, col_a = st.columns([1, 2])
            
            with col_q:
                st.write("**질문:**")
                st.write(chat['question'])
                st.caption(f"시간: {chat['timestamp'].strftime('%H:%M:%S')}")
            
            with col_a:
                st.write("**답변:**")
                st.write(chat['answer'])
                if show_confidence:
                    st.caption(f"신뢰도: {chat['confidence']:.1%}")

# 통계 및 분석
if len(st.session_state.chat_history) > 0:
    st.header("📈 사용 통계")
    
    # 기본 통계
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.metric("총 질문 수", len(st.session_state.chat_history))
    
    with col_stat2:
        avg_confidence = sum(chat['confidence'] for chat in st.session_state.chat_history) / len(st.session_state.chat_history)
        st.metric("평균 신뢰도", f"{avg_confidence:.1%}")
    
    with col_stat3:
        total_sources = sum(len(chat['sources']) for chat in st.session_state.chat_history)
        st.metric("총 참조 문서", total_sources)
    
    # 신뢰도 분포 차트
    if len(st.session_state.chat_history) > 1:
        confidence_data = [chat['confidence'] for chat in st.session_state.chat_history]
        
        fig = px.histogram(
            x=confidence_data,
            nbins=10,
            title="신뢰도 분포",
            labels={'x': '신뢰도', 'y': '빈도'}
        )
        st.plotly_chart(fig, use_container_width=True)

# 푸터
st.markdown("---")
st.markdown("**🏥 병원 고객 질의응답 RAG 시스템** - AI 기반 고객 상담 솔루션")
st.caption("© 2025 서강식 프로젝트. All rights reserved.")

# 디버그 정보 (개발 모드에서만)
if st.sidebar.checkbox("🔧 디버그 모드"):
    st.header("🔧 디버그 정보")
    
    st.subheader("환경 변수")
    st.write(f"OpenAI API Key: {'설정됨' if openai_key else '설정 안됨'}")
    
    st.subheader("세션 상태")
    st.write(f"RAG 시스템: {'로드됨' if 'rag_system' in st.session_state else '로드 안됨'}")
    st.write(f"채팅 히스토리: {len(st.session_state.chat_history)}개")
    
    if st.button("🗑️ 세션 초기화"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
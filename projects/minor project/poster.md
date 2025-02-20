## Heading
- project title : Smart classroom analytics: AI-driven attendance, attentiveness, and lecture management system
- Presented by: Manodeep ray , Harshit , Hasan Iqbal Khan,  Nazheef Biswas
- electronics and computer science engineering
- Guided by: Wriddhi Bhowmik & Prof. B. P. de
- group no . ECSc-92



## Introduction

- **Project Overview**:
    
    - Proposes an AI-driven smart classroom system using Large Language Models (LLMs) and computer vision (CV) for automation and engagement.
- **Motivation**:
    
    - Addresses challenges like manual attendance, attentiveness monitoring, and inefficient note-taking to enhance learning.
- **Background**:
    
    - Advances in YOLO , yunet for CV and LLMs for NLP enable real-time tracking, dynamic information retrieval, and AI-driven insights.
- **Key Features**:
    
    - Uses edge computing for CV tasks and cloud storage for scalability.
    - Custom YOLO and yunet pretrained model for attendance and attentiveness tracking.
    - LLM-powered RAG system for note generation and queries.
- **Impact**:
    
    - Boosts classroom efficiency, provides AI-driven analytics, and enhances learning resources.




## Methodology

- **Face Capture and Recognition**:
    
    - The  Yunet CV model captures student faces using edge devices.
        
    - YOLO is used to recognize faces for attendance and attentiveness monitoring.
        
- **Data Processing and Storage**:
    
    - Captured data is sent to a database for storage and processing.
        
    - A cloud-based dashboard visualizes analytics for teachers, providing insights on attendance and engagement.
        
- **Lecture Recording and Transcription**:
    
    - A functionality records lectures and transcribes them.
        
    - Transcribed data is processed using an LLM to generate class notes.
        
- **Accessibility**:
    
    - Generated notes are accessible to both students and teachers, promoting collaborative learning and review.




## Objectives

- Develop an AI-powered smart classroom system integrating CV and NLP technologies to automate attendance and attentiveness tracking.
    
- Implement a cloud-based dashboard for real-time analytics accessible to students and teachers.
    
- Utilize LLMs for Retrieval-Augmented Generation (RAG) to support dynamic note-taking and context-aware queries.
    
- Enhance learning experiences by providing scalable and efficient AI-driven insights.




## Societal Impact

- **Education Sector**: Enhances learning efficiency, promotes personalized education, and simplifies administrative tasks.
    
- **Industry Applications**: Demonstrates scalable AI solutions for workplace monitoring and training systems.
    
- **Research Community**: Encourages advancements in AI, NLP, and CV integration, fostering further innovations.
    
- **Environmental Impact**: Reduces paper usage through digital note-taking, promoting eco-friendly practices.
    
- **Social Development**: Provides equitable access to quality education, benefiting remote and under-resourced regions.



## Conclusions

This project demonstrates how AI-driven technologies, including CV and NLP, can automate and enhance classroom management. By integrating edge computing and cloud solutions, the system efficiently tracks attendance, monitors attentiveness, and generates class notes. While the current implementation focuses on basic functionalities, its scalability provides opportunities for further development. Constraints include dependency on internet connectivity for cloud operations and hardware capabilities of edge devices.

## Future Work

Future enhancements may include expanding the AI model to support multiple camera angles for better coverage, integrating sentiment analysis for deeper engagement insights, and refining the transcription process for higher accuracy. Additional research can explore multilingual support and adaptive learning modules to further personalize education.

## References

- Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement.
    
- Vaswani, A., et al. (2017). Attention is All You Need.
    
- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
    
- Additional references to be added based on further studies.


yu net https://github.com/geaxgx/depthai_yunet/blob/main/YuNet.py


https://realcoderz.com/skillaTracker/CCTV-class-room-attendance

https://github.com/yash-choudhary/Automatic_attendance_system



quantization of yolo model
https://medium.com/@sulavstha007/quantizing-yolo-v8-models-34c39a2c10e2
https://github.com/majipa007/Quantization-YOLOv8/blob/main/Quantization.ipynb


https://github.com/majipa007/Quantization-YOLOv8/blob/main/requirements.txt
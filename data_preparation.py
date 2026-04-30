import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

def clean_text_simple(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return " ".join(text.split())

def extract_manual_features(text, categories):
    t = text.lower()
    skills = ['python', 'java', 'sql', 'machine learning', 'deep learning', 'nlp', 'c++', 'javascript', 'html', 'css', 'excel', 'management', 'leadership', 'communication', 'sales', 'marketing', 'project management', 'accounting', 'finance', 'design', 'autocad', 'aws', 'azure', 'docker', 'kubernetes', 'react', 'node', 'django', 'flask']
    s_count = sum([1 for s in skills if s in t])
    exp = min(int(re.search(r'(\d+)\+?\s*(?:years|yrs)', t).group(1)) if re.search(r'(\d+)\+?\s*(?:years|yrs)', t) else 0, 30)
    edu = 1 if 'bachelor' in t else 2 if 'master' in t else 3 if 'phd' in t else 0
    
    # 85% Honest Push: Surgical Signals
    dk = {
        'ACCOUNTANT': ['accountant', 'ledger', 'audit', 'taxation', 'tally', 'balance sheet'],
        'ADVOCATE': ['litigation', 'court', 'case law', 'lawyer', 'attorney', 'legal counsel', 'juris'],
        'AGRICULTURE': ['agriculture', 'farm', 'agronomy', 'crop', 'soil'],
        'APPAREL': ['apparel', 'fashion', 'merchandiser', 'garment', 'textile'],
        'ARTS': ['fine art', 'exhibition', 'museum', 'gallery', 'sculpture', 'painting'],
        'AUTOMOBILE': ['automobile', 'engine', 'chassis', 'vehicle', 'automotive'],
        'AVIATION': ['aviation', 'aircraft', 'pilot', 'flight', 'cabin crew'],
        'BANKING': ['banking', 'loan', 'credit', 'branch', 'teller'],
        'BPO': ['call center', 'inbound', 'outbound', 'customer support'],
        'BUSINESS-DEVELOPMENT': ['business development', 'bde', 'lead generation', 'client acquisition'],
        'CHEF': ['chef', 'kitchen', 'culinary', 'cuisine', 'restaurant'],
        'CONSTRUCTION': ['civil engineer', 'site supervisor', 'infrastructure', 'building'],
        'CONSULTANT': ['consultant', 'advisory', 'business analyst', 'strategy'],
        'DESIGNER': ['graphic designer', 'ui ux', 'photoshop', 'illustrator'],
        'DIGITAL-MEDIA': ['seo', 'sem', 'digital marketing', 'content writer', 'social media'],
        'ENGINEERING': ['mechanical engineer', 'electrical engineer', 'maintenance'],
        'FINANCE': ['financial analyst', 'investment banking', 'portfolio'],
        'FITNESS': ['gym', 'yoga', 'trainer', 'health', 'fitness'],
        'HEALTHCARE': ['healthcare', 'medical', 'hospital', 'nurse', 'doctor', 'patient care'],
        'HR': ['human resources', 'recruitment', 'payroll', 'onboarding'],
        'INFORMATION-TECHNOLOGY': ['software developer', 'full stack', 'cloud architect', 'devops', 'backend', 'frontend'],
        'PUBLIC-RELATIONS': ['public relations', 'press release', 'journalism', 'branding'],
        'SALES': ['sales executive', 'retail manager', 'client relationship', 'selling'],
        'TEACHER': ['teacher', 'teaching', 'school', 'professor', 'academic']
    }
    # Increased weight for these surgical signals
    signals = [sum([t.count(k) * 2 for k in dk.get(cat, [])]) for cat in categories]
    return [s_count, exp, edu, len(t.split())] + signals

def load_data_raw(csv_path):
    print("Loading raw dataset...")
    df = pd.read_csv(csv_path)
    return df

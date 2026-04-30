import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy, json, sys, re, torch, joblib, os
import torch.nn as nn
import numpy as np
from company_list import is_company

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ══════════════════════════════════════════════════════════════════
# COMPREHENSIVE SKILL DATABASES
# ══════════════════════════════════════════════════════════════════

TECHNICAL_SKILLS = [
    # ── Programming Languages ─────────────────────────────────────────
    'python', 'java', 'javascript', 'typescript', 'c', 'c++', 'c#', 'ruby', 'go', 'golang',
    'rust', 'swift', 'kotlin', 'scala', 'perl', 'php', 'r', 'matlab', 'dart', 'lua',
    'haskell', 'elixir', 'clojure', 'objective-c', 'assembly', 'fortran', 'cobol',
    'visual basic', 'vba', 'groovy', 'julia', 'solidity', 'verilog', 'vhdl',
    'f#', 'erlang', 'ocaml', 'prolog', 'lisp', 'racket', 'tcl', 'awk',
    'abap', 'apex', 'powershell', 'bash', 'shell', 'shell scripting', 'bash scripting', 'bash/shell', 'zsh',
    'coffeescript', 'elm', 'nim', 'zig', 'crystal', 'd', 'ada', 'pascal',

    # ── Web Frontend ─────────────────────────────────────────────────
    'html', 'html5', 'css', 'css3', 'sass', 'scss', 'less', 'stylus',
    'bootstrap', 'tailwind', 'tailwind css', 'tailwindcss', 'bulma', 'foundation', 'materialize',
    'react', 'reactjs', 'react.js', 'redux', 'recoil', 'zustand', 'mobx',
    'angular', 'angularjs', 'ngrx', 'rxjs',
    'vue', 'vuejs', 'vue.js', 'vuex', 'pinia', 'nuxt', 'nuxtjs',
    'svelte', 'sveltekit', 'next.js', 'nextjs', 'gatsby', 'remix', 'astro',
    'jquery', 'ajax', 'axios', 'fetch api', 'websocket', 'websockets', 'webrtc',
    'webpack', 'vite', 'babel', 'parcel', 'rollup', 'esbuild', 'gulp', 'grunt',
    'storybook', 'pwa', 'progressive web app', 'webassembly',
    'three.js', 'webgl', 'd3.js', 'd3', 'chart.js', 'recharts',
    'material ui', 'ant design', 'chakra ui', 'shadcn', 'headless ui',
    'framer motion', 'gsap',

    # ── Web Backend ───────────────────────────────────────────────────
    'node', 'nodejs', 'node.js', 'express', 'expressjs', 'express.js', 'fastify', 'nestjs', 'koa',
    'django', 'flask', 'fastapi', 'tornado', 'starlette', 'sanic',
    'spring', 'spring boot', 'springboot', 'spring mvc', 'spring security', 'hibernate',
    'asp.net', '.net', 'dotnet', '.net core', 'blazor',
    'rails', 'ruby on rails', 'sinatra', 'laravel', 'symfony', 'codeigniter',
    'gin', 'fiber', 'echo', 'beego',
    'graphql', 'apollo', 'hasura', 'rest', 'restful', 'rest api', 'soap',
    'grpc', 'protobuf', 'thrift',
    'microservices', 'serverless', 'lambda', 'api gateway', 'webhooks',
    'oauth', 'oauth2', 'jwt', 'saml', 'sso',
    'json', 'xml', 'yaml', 'toml',

    # ── Mobile Development ────────────────────────────────────────────
    'android', 'android (kotlin/java)', 'ios', 'ios (swift)', 'flutter', 'react native', 'xamarin', 'ionic',
    'swiftui', 'jetpack compose', 'cordova', 'phonegap', 'capacitor',
    'kotlin multiplatform', 'kmp', 'expo',
    'firebase', 'push notifications', 'fcm', 'apns',

    # ── Databases ─────────────────────────────────────────────────────
    'sql', 'mysql', 'postgresql', 'postgres', 'sqlite', 'oracle', 'oracle db',
    'mongodb', 'redis', 'cassandra', 'dynamodb', 'couchdb', 'neo4j', 'mariadb',
    'firestore', 'supabase', 'elasticsearch', 'opensearch', 'influxdb',
    'mssql', 'sql server', 'microsoft sql server', 'plsql', 'pl/sql', 'nosql', 'hbase', 'cockroachdb',
    'timescaledb', 'arangodb', 'couchbase', 'fauna', 'planetscale',
    'clickhouse', 'druid', 'presto', 'trino',
    'db2', 'teradata', 'greenplum',
    'memcached', 'hazelcast',
    'database design', 'database administration', 'query optimization',
    'stored procedures', 'indexing', 'orm',
    'prisma', 'sequelize', 'typeorm', 'sqlalchemy', 'mongoose',

    # ── Cloud & DevOps ────────────────────────────────────────────────
    'aws', 'amazon web services', 'ec2', 's3', 'rds', 'cloudwatch',
    'azure', 'microsoft azure', 'azure devops', 'azure functions', 'aks',
    'gcp', 'google cloud', 'google cloud platform (gcp)', 'google cloud platform', 'gke', 'cloud run', 'bigquery', 'cloud storage',
    'docker', 'docker compose', 'kubernetes', 'k8s', 'helm', 'kustomize',
    'terraform', 'ansible', 'puppet', 'chef', 'pulumi', 'cloudformation',
    'jenkins', 'gitlab ci', 'github actions', 'circleci', 'travis ci',
    'teamcity', 'bamboo', 'argocd', 'fluxcd',
    'nginx', 'apache', 'caddy', 'haproxy', 'traefik',
    'linux', 'ubuntu', 'centos', 'rhel', 'debian',
    'unix', 'bash',
    'ci/cd', 'cicd', 'devops', 'sre', 'platform engineering',
    'heroku', 'netlify', 'vercel', 'digitalocean', 'linode', 'vagrant',
    'prometheus', 'grafana', 'elk stack', 'logstash', 'kibana', 'fluentd',
    'openshift', 'rancher', 'istio', 'consul', 'vault', 'linkerd',
    'cloud native', 'infrastructure as code', 'iac',
    'load balancer', 'load balancing', 'cdn', 'cloudflare', 'akamai',
    'new relic', 'datadog', 'dynatrace', 'splunk', 'appdynamics',
    'opentelemetry', 'monitoring', 'observability',

    # ── Data Science & AI / ML ────────────────────────────────────────
    'machine learning', 'deep learning', 'artificial intelligence', 'ai/ml',
    'neural networks', 'cnn', 'rnn', 'lstm', 'transformer', 'attention mechanism',
    'nlp', 'natural language processing', 'computer vision', 'object detection',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'scikit', 'sklearn',
    'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'bokeh',
    'opencv', 'pillow', 'spacy', 'nltk', 'gensim', 'textblob',
    'hugging face', 'transformers', 'diffusers', 'accelerate', 'peft', 'lora',
    'xgboost', 'lightgbm', 'catboost', 'random forest', 'gradient boosting',
    'svm', 'logistic regression', 'linear regression', 'k-means', 'dbscan',
    'pca', 'tsne', 'umap',
    'regression', 'classification', 'clustering', 'anomaly detection',
    'time series', 'arima', 'prophet',
    'reinforcement learning', 'q-learning',
    'generative ai', 'gpt', 'llm', 'large language models', 'rag',
    'prompt engineering', 'fine-tuning', 'instruction tuning', 'rlhf',
    'langchain', 'llamaindex', 'semantic search', 'embeddings',
    'vector database', 'pinecone', 'weaviate', 'chromadb', 'qdrant', 'faiss', 'milvus',
    'mlflow', 'kubeflow', 'mlops', 'bentoml',
    'feature engineering', 'feature store', 'model deployment', 'model serving',
    'a/b testing', 'experiment tracking', 'weights & biases', 'wandb', 'dvc',
    'stable diffusion', 'openai api', 'ollama',

    # ── Data Engineering & Analytics ─────────────────────────────────
    'data analytics', 'data analysis', 'data engineering', 'data pipeline',
    'etl', 'elt', 'data warehousing', 'data modeling', 'data governance',
    'tableau', 'power bi', 'looker', 'metabase', 'qlik', 'qlikview', 'qliksense',
    'superset', 'redash', 'sigma', 'google data studio',
    'apache spark', 'spark', 'pyspark', 'hadoop', 'hive', 'pig', 'mapreduce',
    'airflow', 'apache airflow', 'prefect', 'dagster', 'luigi',
    'kafka', 'apache kafka', 'confluent', 'rabbitmq', 'celery',
    'flink', 'apache flink', 'beam', 'apache beam',
    'snowflake', 'databricks', 'redshift', 'athena', 'azure synapse',
    'delta lake', 'apache iceberg', 'lakehouse',
    'dbt', 'fivetran', 'stitch', 'airbyte', 'talend', 'informatica', 'mulesoft',
    'excel', 'excel (advanced)', 'google sheets', 'pivot tables', 'vlookup', 'macros', 'power query',
    'sas', 'spss', 'stata', 'minitab',
    'data visualization', 'business intelligence',

    # ── Cybersecurity ─────────────────────────────────────────────────
    'cybersecurity', 'information security', 'penetration testing',
    'ethical hacking', 'vulnerability assessment', 'red team', 'blue team',
    'siem', 'soc', 'ids', 'ips', 'firewall', 'firewalls', 'waf', 'dlp',
    'encryption', 'ssl', 'tls', 'pki', 'cryptography',
    'owasp', 'burp suite', 'metasploit', 'nmap', 'wireshark', 'kali linux',
    'nessus', 'qualys', 'snort', 'suricata',
    'iso 27001', 'nist', 'soc2', 'gdpr', 'hipaa', 'sox', 'pci dss',
    'identity management', 'iam', 'active directory', 'ldap', 'azure ad',
    'zero trust', 'devsecops', 'threat modeling', 'incident response',

    # ── Version Control & Collaboration ──────────────────────────────
    'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial', 'perforce',
    'jira', 'confluence', 'trello', 'asana', 'notion', 'monday.com', 'linear',
    'slack', 'microsoft teams', 'zoom', 'miro',
    'code review', 'pull requests', 'gitflow',
    'postman',

    # ── Testing & QA ─────────────────────────────────────────────────
    'unit testing', 'integration testing', 'end-to-end testing',
    'selenium', 'cypress', 'playwright', 'puppeteer',
    'jest', 'mocha', 'chai', 'jasmine', 'vitest',
    'pytest', 'unittest', 'hypothesis',
    'junit', 'testng', 'mockito',
    'cucumber', 'gherkin', 'behave',
    'swagger', 'insomnia',
    'load testing', 'jmeter', 'gatling', 'locust', 'k6',
    'qa', 'quality assurance', 'tdd', 'bdd', 'test automation',
    'regression testing', 'smoke testing', 'exploratory testing',
    'appium', 'robot framework', 'katalon',
    'sonarqube', 'code coverage',

    # ── Design & Creative ─────────────────────────────────────────────
    'photoshop', 'illustrator', 'figma', 'sketch', 'adobe xd', 'invision',
    'zeplin', 'framer', 'protopie', 'balsamiq',
    'premiere pro', 'after effects', 'davinci', 'resolve', 'davinci resolve',
    'final cut pro', 'avid', 'lightroom', 'indesign', 'canva',
    'blender', 'maya', '3ds max', 'cinema 4d', 'houdini', 'zbrush',
    'unity', 'unity (game dev)', 'unreal engine', 'godot',
    'ui/ux', 'ui design', 'ux design', 'wireframing', 'prototyping',
    'user research', 'usability testing',
    'responsive design', 'material design', 'design systems',
    'video editing', 'non-linear editing', 'motion graphics', 'animation', 'graphic design',
    'typography', 'branding', 'identity design',
    'hootsuite', 'buffer', 'sprout social',
    'adobe creative suite', 'adobe creative cloud',

    # ── Networking & Infrastructure ───────────────────────────────────
    'tcp/ip', 'dns', 'dhcp',
    'vpn', 'lan', 'wan', 'sdwan', 'mpls', 'bgp', 'ospf',
    'osi model', 'subnetting', 'cisco networking',
    'cisco', 'juniper', 'palo alto', 'fortinet', 'checkpoint',
    'vmware', 'vsphere', 'vcenter', 'hyper-v', 'virtualbox', 'proxmox',
    'network security', 'network monitoring', 'sdn',
    'nagios', 'zabbix',

    # ── ERP & Enterprise Systems ──────────────────────────────────────
    'sap', 'sap hana', 'sap fico', 'sap mm', 'sap sd', 'sap abap',
    'sap s/4hana', 'sap bw', 'sap crm', 'sap ariba',
    'oracle erp', 'oracle fusion', 'peoplesoft',
    'workday', 'workday hcm',
    'salesforce', 'salesforce crm', 'soql',
    'servicenow', 'zendesk', 'hubspot', 'freshdesk',
    'dynamics 365', 'power apps', 'power automate', 'power platform',
    'netsuite', 'sage', 'tally', 'zoho crm', 'odoo',

    # ── Blockchain & Web3 ─────────────────────────────────────────────
    'blockchain', 'ethereum', 'solidity', 'web3', 'web3.js', 'ethers.js',
    'smart contracts', 'defi', 'nft', 'ipfs', 'hyperledger',
    'bitcoin', 'polygon', 'solana', 'hardhat', 'truffle', 'foundry', 'chainlink',

    # ── IDEs & Developer Tools ────────────────────────────────────────
    'vs code', 'visual studio', 'intellij', 'pycharm', 'webstorm', 'goland',
    'eclipse', 'netbeans', 'xcode', 'android studio', 'sublime text', 'vim', 'nvim',
    'emacs', 'jupyter', 'jupyter notebook', 'jupyterlab', 'google colab', 'anaconda',
    'cursor', 'replit',

    # ── Software Architecture & Patterns ──────────────────────────────
    'data structures & algorithms', 'data structures', 'algorithms', 'oop', 'system design',
    'microservices', 'event driven', 'cqrs', 'event sourcing',
    'design patterns', 'solid principles', 'clean code', 'clean architecture',
    'domain driven design', 'ddd', 'api design',
    'distributed systems', 'caching', 'message queues',
    'rate limiting', 'circuit breaker',

    # ── Agile & Project Methodologies ────────────────────────────────
    'agile', 'scrum', 'kanban', 'waterfall', 'lean', 'safe', 'xp',
    'sprint planning', 'retrospectives',
    'six sigma', 'itil', 'prince2', 'pmp',

    # ── Hardware, Embedded & IoT ──────────────────────────────────────
    'iot', 'embedded systems', 'arduino', 'raspberry pi', 'esp32', 'esp8266',
    'plc', 'rtos', 'freertos', 'stm32', 'microcontrollers', 'fpga',
    'uart', 'spi', 'i2c', 'can bus', 'modbus',
    'mqtt', 'zigbee', 'lora', 'bluetooth',
    'robotics', 'ros', 'ros (robotics)', 'ros2',
    'autocad', 'solidworks', 'catia', 'ansys', 'simulink', 'labview',

    # ── CMS, E-commerce & Digital Marketing ──────────────────────────
    'wordpress', 'drupal', 'joomla', 'ghost', 'contentful', 'strapi', 'sanity',
    'magento', 'shopify', 'woocommerce', 'prestashop', 'bigcommerce',
    'google analytics', 'ga4', 'google tag manager', 'mixpanel', 'amplitude',
    'seo', 'sem', 'google ads', 'facebook ads', 'meta ads',
    'mailchimp', 'klaviyo', 'marketo', 'pardot',
    'ahrefs', 'semrush', 'moz',
    'crm', 'cms', 'erp',

    # ── Finance & Quant ───────────────────────────────────────────────
    'algorithmic trading', 'quantitative analysis', 'financial modelling',
    'bloomberg terminal', 'refinitiv', 'factset',
    'risk management', 'monte carlo simulation',
    'portfolio optimization', 'backtesting',
    'excel vba', 'financial reporting', 'gaap', 'ifrs',
]

SOFT_SKILLS = [
    # Leadership & Management
    'leadership', 'team leadership', 'team management', 'people management',
    'strategic planning', 'strategic thinking', 'decision making', 'delegation',
    'mentoring', 'coaching', 'talent development', 'performance management',
    'change management', 'conflict resolution', 'stakeholder management',
    'cross-functional', 'cross functional collaboration', 'vision setting',

    # Communication
    'communication', 'verbal communication', 'written communication',
    'public speaking', 'presentation skills', 'presentations',
    'storytelling', 'active listening', 'negotiation', 'persuasion',
    'interpersonal skills', 'client relations', 'client management',
    'technical writing', 'documentation', 'report writing', 'proposal writing',
    'email etiquette',

    # Problem Solving & Analytical
    'problem solving', 'critical thinking', 'analytical thinking', 'analytical skills',
    'root cause analysis', 'troubleshooting', 'debugging', 'research', 'research skills',
    'data-driven', 'data driven decision making', 'data-driven decision making', 'logical thinking',
    'creative thinking', 'innovation', 'design thinking', 'creative problem solving', 'logical reasoning',
    'brainstorming', 'ideation', 'lateral thinking',

    # Teamwork & Collaboration
    'teamwork', 'collaboration', 'team player', 'cooperative',
    'relationship building', 'networking', 'empathy',
    'cultural awareness', 'diversity', 'inclusion',
    'remote collaboration', 'virtual teams',
    'emotional intelligence', 'emotional intelligence (eq)', 'eq', 'diplomacy',

    # Organization & Productivity
    'time management', 'multitasking', 'prioritization', 'organization',
    'attention to detail', 'detail oriented', 'deadline driven',
    'goal setting', 'planning', 'scheduling', 'resource management',
    'project management', 'program management', 'portfolio management',
    'risk management', 'budget management', 'cost management',
    'process improvement', 'process optimization', 'workflow optimization',

    # Adaptability & Growth
    'adaptability', 'flexibility', 'resilience', 'growth mindset',
    'self-motivated', 'self motivated', 'initiative', 'proactive',
    'fast learner', 'quick learner', 'continuous learning', 'fast learning',
    'open minded', 'receptive to feedback', 'open to feedback', 'self-awareness',
    'stress management', 'handling ambiguity',

    # Business & Professional
    'customer service', 'customer focus', 'customer satisfaction',
    'sales', 'business development', 'business acumen',
    'entrepreneurship', 'startup mindset',
    'financial literacy', 'budgeting', 'forecasting',
    'market research', 'competitive analysis', 'swot analysis',
    'brand management', 'digital marketing', 'content marketing', 'content creation',
    'account management', 'vendor management', 'supply chain',
    'quality control', 'quality management', 'compliance',
    'training', 'facilitation', 'onboarding', 'knowledge transfer',
    'event planning', 'event management',

    # Work Ethics
    'work ethic', 'professionalism', 'accountability', 'integrity',
    'reliability', 'dependability', 'punctuality',
    'confidentiality', 'ethics', 'transparency',
    'patience', 'perseverance', 'tenacity', 'discipline',
    'work under pressure', 'deadline management',

    # Customer & Stakeholder Focus
    'stakeholder communication', 'requirements gathering', 'user empathy', 'expectation management',
]

class NERPipeline:
    def __init__(self, model_name="dslim/bert-base-NER"):
        print("Activating Clean Recruiter NER...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="max")
        
        # Load Job Classifier Components
        try:
            self.tfidf = joblib.load('tfidf_vectorizer.joblib')
            self.le = joblib.load('label_encoder.joblib')
            self.scaler = joblib.load('scaler.joblib')
            self.selector = joblib.load('selector.joblib')
            
            with open('model_config.json', 'r') as f:
                config = json.load(f)
            
            from train_dl import ResumeMLP
            self.clf = ResumeMLP(config['input_size'], config['num_classes'])
            self.clf.load_state_dict(torch.load('resume_mlp_model.pth', map_location='cpu'))
            self.clf.eval()
            self.has_clf = True
            print("Job Classifier Integrated!")
        except Exception as e:
            print(f"Job Classifier not loaded: {e}")
            self.has_clf = False

    def extract_text(self, pdf_path):
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc: text += page.get_text()
        except: pass
        return text

    def scan_skills(self, text_lower):
        """Scan resume text and return categorized skills with precision filtering."""
        raw_tech = []
        raw_soft = []
        
        # Words that are technical skills but often appear as common English words
        # in non-tech resumes. We ignore these if they are the ONLY match.
        SKILL_NOISE = ['linear', 'resolve', 'avid', 'lead', 'direct', 'impact', 'read', 'go', 'dart']
        
        # Initial capture
        for skill in TECHNICAL_SKILLS:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                # Strict check for single-letter skills - usually junk in resumes
                if len(skill) == 1: continue 
                raw_tech.append(skill.title())
        
        for skill in SOFT_SKILLS:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                raw_soft.append(skill.title())

        # Deduplication Logic: Only keep the longest version of overlapping skills
        # e.g., keep "Davinci Resolve", remove "Davinci"
        def deduplicate(skills):
            final = []
            # Sort by length descending to ensure we check longer phrases first
            sorted_skills = sorted(list(set(skills)), key=len, reverse=True)
            for skill in sorted_skills:
                # Normalize for comparison (handles hyphens like non-linear vs linear)
                norm_skill = skill.lower().replace('-', ' ')
                
                # If this skill is already represented by a longer phrase in 'final', skip it
                is_subset = False
                for existing in final:
                    norm_existing = existing.lower().replace('-', ' ')
                    if norm_skill in norm_existing:
                        is_subset = True
                        break
                
                if not is_subset:
                    # Also skip noise words if they are too short and context is weak
                    if skill.lower() in SKILL_NOISE and len(skill) < 8:
                        if len(sorted_skills) < 3: continue
                    final.append(skill)
            return final

        return sorted(deduplicate(raw_tech)), sorted(deduplicate(raw_soft))

    def process(self, pdf_path):
        text = self.extract_text(pdf_path)
        if not text: return {"error": "Could not read PDF"}
        raw_lower = text.lower()
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        # 1. Candidate Name (Header Filtering + Email Fallback)
        candidate_name = "Unknown"
        headers = ["resume", "cv", "curriculum", "vitae", "undergraduate", "profile", "linkedin", "github", "education", "skills"]
        for line in lines[:8]:
            line_low = line.lower()
            if not any(h in line_low for h in headers):
                clean_l = re.sub(r'[^a-zA-Z\s]', '', line).strip()
                if 1 <= len(clean_l.split()) <= 4 and len(clean_l) > 2:
                    if "http" not in line_low and ":" not in line_low:
                        candidate_name = clean_l
                        break
        
        if candidate_name == "Unknown" or any(h in candidate_name.lower() for h in headers):
            email_match = re.search(r'([a-zA-Z0-9._%+-]+)@', text)
            if email_match:
                raw_name = email_match.group(1)
                candidate_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', raw_name).replace('.', ' ').title()

        # 2. Skill Scan
        tech_skills, soft_skills = self.scan_skills(raw_lower)

        # 3. Companies & Roles
        doc = nlp(text[:3000])
        NOT_COMPANIES = {
            'microsoft', 'google', 'amazon', 'aws', 'azure', 'oracle', 'cisco',
            'adobe', 'apple', 'meta', 'facebook', 'docker', 'kubernetes', 'git',
            'github', 'linux', 'windows', 'android', 'ios', 'excel', 'word',
            'powerpoint', 'office', 'suite', 'workspace', 'cloud', 'linkedin',
            'google ads', 'github.com', 'linkedin.com', 'ssl'
        }

        companies = []
        roles = []

        # BERT NER for Entities
        entities = self.ner_pipeline(text[:2500])
        for ent in entities:
            word = ent['word'].replace('##', '').strip()
            word_low = word.lower()
            if ent['entity_group'] == 'ORG' and is_company(word) and word_low not in NOT_COMPANIES:
                companies.append(word)
        
        # Fallback Company Scanner (Scan text directly against COMPANY_DB)
        from company_list import COMPANY_DB_LOWER
        for company in COMPANY_DB_LOWER:
            if len(company) > 3 and company.lower() not in NOT_COMPANIES:
                if re.search(r'\b' + re.escape(company) + r'\b', raw_lower):
                    companies.append(company.title())

        # Role Discovery (Expanded Database)
        job_titles = [
            "Frontend Developer", "Backend Developer", "Full Stack Developer", "Web Developer", 
            "Software Engineer", "Software Developer", "Junior Developer", "Senior Developer", 
            "Lead Developer", "Principal Engineer", "Staff Engineer", "Distinguished Engineer", 
            "Mobile Developer", "Android Developer", "iOS Developer", "React Developer", 
            "Node.js Developer", "PHP Developer", ".NET Developer", "Java Developer", 
            "Python Developer", "Ruby on Rails Developer", "Embedded Systems Engineer", 
            "Firmware Engineer", "Game Developer", "Unity Developer", "Unreal Engine Developer",
            "Machine Learning Engineer", "Deep Learning Engineer", "AI Engineer", "Applied Scientist", 
            "Research Scientist", "NLP Engineer", "Computer Vision Engineer", "Data Scientist", 
            "Junior Data Scientist", "Senior Data Scientist", "Lead Data Scientist", 
            "AI Research Scientist", "MLOps Engineer", "AI Product Manager", "Prompt Engineer", 
            "LLM Engineer", "Generative AI Engineer", "AI Safety Researcher", "Reinforcement Learning Engineer", 
            "AI Ethicist", "Data Analyst", "Senior Data Analyst", "Business Analyst", "Data Engineer", 
            "Senior Data Engineer", "Data Architect", "Analytics Engineer", "BI Developer", 
            "BI Analyst", "Business Intelligence Analyst", "Reporting Analyst", "Database Administrator", 
            "Database Engineer", "ETL Developer", "Data Warehouse Engineer", "Big Data Engineer", 
            "Quantitative Analyst", "Statistical Analyst", "Research Analyst", "DevOps Engineer", 
            "Senior DevOps Engineer", "Site Reliability Engineer", "SRE", "Platform Engineer", 
            "Cloud Engineer", "Cloud Architect", "AWS Engineer", "Azure Engineer", "GCP Engineer", 
            "Infrastructure Engineer", "Systems Engineer", "Systems Administrator", "Linux Administrator", 
            "Network Engineer", "Network Administrator", "Storage Engineer", "Cloud Security Engineer", 
            "Release Engineer", "Build Engineer", "Automation Engineer", "Cybersecurity Analyst", 
            "Information Security Analyst", "Security Engineer", "Security Architect", "Penetration Tester", 
            "Ethical Hacker", "Red Team Engineer", "Blue Team Engineer", "SOC Analyst", "Threat Intelligence Analyst", 
            "Incident Response Analyst", "Malware Analyst", "Forensics Analyst", "Application Security Engineer", 
            "Vulnerability Researcher", "CISO", "Security Consultant", "GRC Analyst", "Compliance Analyst",
            "QA Engineer", "QA Analyst", "Software Tester", "Manual Tester", "Automation Test Engineer", 
            "SDET", "Performance Test Engineer", "Test Lead", "Test Manager", "QA Manager", 
            "API Test Engineer", "Mobile Test Engineer", "Accessibility Tester", "Software Architect", 
            "Solution Architect", "Enterprise Architect", "Technical Architect", "CTO", "VP of Engineering", 
            "Director of Engineering", "Head of Technology", "Engineering Manager", "Technical Lead", "Tech Lead",
            "IT Support Engineer", "IT Technician", "Help Desk Analyst", "Service Desk Analyst", 
            "Desktop Support Engineer", "Field Service Engineer", "IT Manager", "IT Director", 
            "IT Coordinator", "System Administrator", "IT Operations Manager", "ERP Consultant", 
            "SAP Consultant", "IT Auditor", "IT Procurement Specialist", "Product Manager", 
            "Senior Product Manager", "Associate Product Manager", "Principal Product Manager", 
            "Group Product Manager", "Director of Product", "VP of Product", "Chief Product Officer", 
            "Product Owner", "UX Designer", "UI Designer", "UX Researcher", "Product Designer", 
            "Visual Designer", "Interaction Designer", "Motion Designer", "Design Lead", "Head of Design", 
            "UX Writer", "Content Designer", "Design Systems Engineer", "Creative Director", "Art Director",
            "Strategy Consultant", "Management Consultant", "Operations Manager", "Business Development Manager", 
            "Corporate Strategist", "Financial Analyst", "Investment Analyst", "Equity Research Analyst", 
            "Risk Analyst", "Credit Analyst", "Actuary", "Financial Planner", "Portfolio Manager", 
            "Hedge Fund Analyst", "Private Equity Analyst", "Chief Financial Officer", "Finance Manager", 
            "Treasury Analyst", "Tax Consultant", "Auditor", "Cost Accountant", "Chartered Accountant", 
            "Budget Analyst", "Economist", "Digital Marketing Specialist", "SEO Specialist", "SEM Specialist", 
            "Content Marketer", "Social Media Manager", "Brand Manager", "Performance Marketer", 
            "Growth Hacker", "Email Marketing Specialist", "Marketing Analyst", "Campaign Manager", 
            "Product Marketing Manager", "Influencer Marketing Manager", "Community Manager", "PR Specialist", 
            "Content Strategist", "Copywriter", "Video Marketing Specialist", "Marketing Automation Specialist", 
            "Chief Marketing Officer", "Sales Executive", "Sales Representative", "Business Development Executive", 
            "Account Manager", "Key Account Manager", "Enterprise Sales Manager", "Inside Sales Representative", 
            "Outside Sales Representative", "Sales Engineer", "Pre-Sales Consultant", "Solutions Consultant", 
            "Channel Sales Manager", "Sales Operations Analyst", "Revenue Operations Manager", "VP of Sales", 
            "Chief Revenue Officer", "HR Executive", "HR Manager", "HR Business Partner", "Talent Acquisition Specialist", 
            "Recruiter", "Technical Recruiter", "Sourcing Specialist", "Compensation Analyst", "L&D Specialist", 
            "Training Manager", "HR Analyst", "Workforce Planning Analyst", "Employee Relations Manager", 
            "OD Consultant", "HRIS Analyst", "Chief People Officer", "DEI Specialist", "Legal Counsel", 
            "Corporate Lawyer", "Compliance Officer", "Regulatory Affairs Specialist", "Contract Manager", 
            "Paralegal", "Intellectual Property Analyst", "Privacy Officer", "DPO", "Legal Operations Manager", 
            "Risk & Compliance Manager", "Employment Lawyer", "Litigation Specialist", "Clinical Data Analyst", 
            "Bioinformatics Scientist", "Medical Data Scientist", "Healthcare IT Specialist", "EHR Consultant", 
            "Clinical Research Coordinator", "Biostatistician", "Pharmacovigilance Specialist", 
            "Regulatory Affairs Manager", "Health Informatician", "Telemedicine Specialist", "Medical Imaging Analyst", 
            "Clinical Systems Analyst", "Mechanical Engineer", "Civil Engineer", "Electrical Engineer", 
            "Electronics Engineer", "Chemical Engineer", "Industrial Engineer", "Production Engineer", 
            "Quality Control Engineer", "Process Engineer", "Manufacturing Engineer", "Robotics Engineer", 
            "Maintenance Engineer", "Structural Engineer", "Aerospace Engineer", "Mechatronics Engineer", 
            "CAD Designer", "Project Engineer", "Research Fellow", "Research Associate", "Postdoctoral Researcher", 
            "Research Engineer", "Principal Investigator", "Professor", "Associate Professor", "Assistant Professor", 
            "Lecturer", "Academic Advisor", "Curriculum Developer", "Instructional Designer", 
            "Education Technology Specialist", "Data Researcher", "Content Writer", "Technical Writer", 
            "Blog Writer", "Scriptwriter", "Journalist", "Editor", "Video Editor", "Graphic Designer", 
            "Animator", "Illustrator", "Podcast Producer", "YouTuber", "Photographer", "Videographer", 
            "Creative Writer", "Documentation Specialist", "Social Media Content Creator", "Storyboard Artist",
            "Operations Analyst", "Supply Chain Analyst", "Logistics Coordinator", "Procurement Specialist", 
            "Vendor Manager", "Warehouse Manager", "Inventory Manager", "Fleet Manager", "Demand Planner", 
            "Category Manager", "Sourcing Manager", "Import/Export Coordinator", "Chief Operating Officer",
            "Banking Analyst", "Loan Officer", "Relationship Manager", "Investment Banker", "Treasury Manager", 
            "Payments Specialist", "Blockchain Developer", "Smart Contract Developer", "FinTech Product Manager", 
            "RegTech Analyst", "Anti-Money Laundering Analyst", "KYC Analyst", "Credit Risk Manager", 
            "Wealth Manager", "Forex Trader", "Intern", "SDE", "Freelancer"
        ]
        
        # 3b. spaCy Dependency Context Validation (Requirement: worked at [ORG])
        doc = nlp(text[:5000])
        validated_companies = []
        context_verbs = ["work", "intern", "join", "employ", "serve", "lead", "manage"]
        for ent in doc.ents:
            if ent.label_ == "ORG" and is_company(ent.text):
                head = ent.root.head
                if any(v in head.lemma_.lower() for v in context_verbs) or any(v in ent.text.lower() for v in ["technologies", "inc", "corp", "solutions"]):
                    validated_companies.append(ent.text.title())
        for c in validated_companies:
            if c not in companies: companies.append(c)

        # 3c. Role Discovery — Regex scan across full text (RESTORED)
        roles = []
        job_titles_sorted = sorted(job_titles, key=len, reverse=True)
        for title in job_titles_sorted:
            pattern = r'\b' + re.escape(title) + r'\b'
            if re.search(pattern, raw_lower, re.IGNORECASE):
                roles.append(title)

        # 4. Predict Job Category (MLP Model)
        predicted_cat = "Unknown"
        if self.has_clf:
            from data_preparation import extract_manual_features, clean_text_simple
            clean_text = clean_text_simple(text)
            tfidf_feat = self.tfidf.transform([clean_text])
            sel_feat = self.selector.transform(tfidf_feat)
            categories = sorted(self.le.classes_)
            man_feat = self.scaler.transform([extract_manual_features(text, categories)])
            X = np.hstack((man_feat, sel_feat.toarray()))
            with torch.no_grad():
                outputs = self.clf(torch.FloatTensor(X))
                _, pred = torch.max(outputs, 1)
                predicted_cat = self.le.inverse_transform([pred.item()])[0]

        # 5. Experience & Duration Logic
        def get_month(m_str):
            m_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
            m_str = m_str.lower()[:3]
            return m_map.get(m_str, 1)

        experience_list = []
        month_names = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
        # Find all month-year pairs independently (more flexible spacing)
        date_hits = []
        hit_pattern = re.compile(rf'({month_names})\s*(?:[\'’\s]+)?(\d{{2,4}})?', re.IGNORECASE)
        for hit in hit_pattern.finditer(raw_lower):
            m = get_month(hit.group(1))
            y = int(hit.group(2)) if hit.group(2) else 2024
            if y < 100: y += 2000
            date_hits.append({'m': m, 'y': y, 'pos': hit.start(), 'text': hit.group(0)})

        # Link pairs that are close together (within 120 chars)
        skip_next = False
        section_blacklist = ["certifications", "achievements", "education", "skills", "projects"]
        
        for i in range(len(date_hits)):
            if skip_next:
                skip_next = False
                continue
                
            start_date = date_hits[i]
            
            # 1. Skip if this date is inside a blacklisted section
            text_around = raw_lower[max(0, start_date['pos']-100):start_date['pos']]
            if any(b in text_around for b in section_blacklist):
                continue

            end_date = None
            # Check for "Present" or "Since"
            following_text = raw_lower[start_date['pos']+len(start_date['text']):start_date['pos']+50]
            if any(x in following_text for x in ["present", "current", "now", "since"]):
                end_m, end_y = 4, 2026 
            elif i + 1 < len(date_hits) and (date_hits[i+1]['pos'] - start_date['pos']) < 150:
                end_date = date_hits[i+1]
                end_m, end_y = end_date['m'], end_date['y']
                skip_next = True
            else: continue

            start_m, start_y = start_date['m'], start_date['y']
            total_months = (end_y - start_y) * 12 + (end_m - start_m)
            if total_months <= 0: total_months = 1
            duration_str = f"{total_months // 12}y {total_months % 12}m" if total_months >= 12 else f"{total_months} months"
            
            # Link to Role/Company (Wider Window)
            line_idx = raw_lower[:start_date['pos']].count('\n')
            context_lines = lines[max(0, line_idx-3):line_idx+3]
            
            found_role = "Unknown"
            # Priority 1: Check our list
            for r in roles:
                if any(r.lower() in line.lower() for line in context_lines):
                    found_role = r
                    break
            # Priority 2: Greedy search for "Developer" or "Intern" if still unknown
            if found_role == "Unknown":
                for line in context_lines:
                    if "developer" in line.lower(): found_role = "Developer"
                    elif "intern" in line.lower(): found_role = "Intern"

            found_company = "Unknown"
            for c in companies:
                if any(c.lower() in line.lower() for line in context_lines):
                    found_company = c
                    break
            
            if found_role != "Unknown" or found_company != "Unknown":
                experience_list.append({
                    "ROLE": found_role,
                    "COMPANY": found_company,
                    "DURATION": duration_str,
                    "PERIOD": f"{start_date['text']} - {end_date['text'] if end_date else 'Present'}".title()
                })

        # Clean duplicates and overlaps (Longest Match Wins)
        def clean_entities(items):
            final = []
            sorted_items = sorted(list(set(items)), key=len, reverse=True)
            for item in sorted_items:
                if not any(item.lower() in other.lower() for other in final):
                    final.append(item)
            return sorted(final)

        # Final Cleanup of special characters in durations/periods
        for exp in experience_list:
            exp['PERIOD'] = exp['PERIOD'].replace('\u2019', "'").replace('\u2013', "-").replace('\u2014', "-")

        # 6. displaCy Visualization (Requirement: Visualize extracted entities)
        from spacy import displacy
        # Create a custom doc for visualization with our extracted entities
        viz_doc = nlp(text)
        # We can also generate a dedicated HTML file
        html = displacy.render(viz_doc, style="ent", page=True)
        with open("entities.html", "w", encoding="utf-8") as f:
            f.write(html)

        return {
            "NAME": candidate_name,
            "PREDICTED_CATEGORY": predicted_cat,
            "EXPERIENCE": experience_list,
            "TECHNICAL_SKILLS": tech_skills,
            "SOFT_SKILLS": soft_skills,
            "COMPANIES": clean_entities(c for c in companies if c.lower() != candidate_name.lower()),
            "ROLES": clean_entities(r for r in roles if r.lower() != candidate_name.lower())
        }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pipe = NERPipeline()
        print(json.dumps(pipe.process(sys.argv[1]), indent=2, ensure_ascii=False))

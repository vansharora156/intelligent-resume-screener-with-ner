"""
company_list.py  —  Worldwide Company Database (Small → Big)
Focus: Companies where candidates WORK, not educational institutions.
"""
import re

COMPANY_DB = {
    # SMALL / BOUTIQUE
    "admec", "admec multimedia", "blue penguin", "blue penguin designs",
    "botle bob", "botle bob advertising", "knoll pharmaceuticals", "knoll",
    "sievert landscaping", "webchutney", "pinstorm", "creativeland asia", 
    "taproot dentsu", "schbang", "social beat", "foxymoron", "mirum india", 
    "blogworks", "stark communications", "brand curry", "langoor", "phoenix media", 
    "creative inc", "the 120 media collective", "whyze group", "spring marketing capital",
    "appinventiv", "hyperlink infosystem", "net solutions", "konstant infosolutions",
    "tatvasoft", "positiwise", "iflexion", "ergonized", "codiant", "brainvire",
    "space-o technologies", "peerbits", "quytech", "openxcell", "techno loader", 
    "sparx it solutions", "softprodigy", "ittisa", "agira technologies", 
    "sphinx solutions", "valuecoders", "bacancy technology", "chetu", "azoft", 
    "instinctools", "emerline", "devox software", "jelvix", "steelkiwi",
    "icreon", "robosoft", "e-zest solutions", "e-zest", "concept pr", 
    "hanmer communications", "mslgroup", "genesis bcw", "adfactors pr",
    "crayons advertising", "enormous brands",

    # IT SERVICES / STAFFING
    "mastech", "mastech holdings", "igate", "patni computer systems",
    "zensar", "zensar technologies", "firstsource", "exlservice", "exl",
    "wns", "wns global services", "genpact", "syntel", "niit technologies", "niit",
    "mphasis", "hexaware", "hexaware technologies", "birlasoft", "persistent systems", 
    "persistent", "cyient", "kpit technologies", "mindtree", "ltimindtree", 
    "l&t infotech", "nucleus software", "newgen software", "newgen", "sonata software", 
    "ramsol", "rsystems", "cigniti", "datamatics", "tata elxsi", "tata communications",
    "3i infotech", "apptad", "infoedge", "naukri",

    # ADVERTISING / MEDIA
    "wpp", "publicis", "omnicom", "interpublic", "dentsu", "havas", "grey group",
    "ogilvy", "bbdo", "ddb", "leo burnett", "saatchi & saatchi", "mccann", "jwt",
    "tbwa", "y&r", "wunderman thompson", "vmly&r", "digitas", "razorfish", "sapient",
    "publicis sapient", "sapientnitro", "contract advertising", "rediffusion",
    "mudra communications", "lintas", "lowe lintas", "ddb mudra", "fcb ulka", 
    "rk swamy bbdo", "rediffusion y&r", "isobar", "dentsu webchutney", "tribal worldwide",
    "the marketing store", "edelman", "edelman india", "mccann worldgroup", "mccann health",

    # INDIA LARGE
    "tcs", "tata consultancy services", "infosys", "wipro", "hcl technologies", "hcl",
    "tech mahindra", "cognizant", "reliance industries", "reliance jio", "jio",
    "reliance retail", "reliance digital", "adani group", "adani enterprises",
    "adani ports", "adani green energy", "tata group", "tata motors", "tata steel",
    "tata chemicals", "tata consumer products", "tata power", "tata projects",
    "mahindra & mahindra", "mahindra", "bajaj auto", "bajaj finserv", "bajaj finance",
    "hero motocorp", "larsen & toubro", "l&t", "l&t technology services", "ltts",
    "l&t construction", "l&t finance", "bhel", "bharat heavy electricals", "ongc", 
    "ioc", "bpcl", "hpcl", "ntpc", "sail", "coal india", "nalco", "hindustan unilever", 
    "hul", "nestle india", "britannia industries", "itc limited", "itc", "dabur india", 
    "marico", "emami", "godrej consumer products", "godrej properties", "colgate palmolive india",
    "asian paints", "berger paints", "kansai nerolac", "pidilite industries", "pidilite",
    "dr reddys laboratories", "sun pharma", "sun pharmaceutical", "cipla", "lupin", 
    "biocon", "aurobindo pharma", "cadila healthcare", "zydus cadila", "zydus",
    "wockhardt", "glenmark", "torrent pharma", "apollo hospitals", "fortis healthcare", 
    "max healthcare", "manipal hospitals", "narayana health", "medanta", "lilavati hospital",
    "kotak mahindra bank", "kotak", "hdfc bank", "hdfc", "hdfc life", "icici bank", 
    "icici", "axis bank", "sbi", "state bank of india", "pnb", "punjab national bank",
    "bank of baroda", "bank of india", "canara bank", "union bank of india", "yes bank", 
    "indusind bank", "federal bank", "idfc first bank", "bandhan bank", "rbl bank", 
    "lic", "general insurance corporation", "paytm", "one97 communications", "phonepe",
    "razorpay", "cashfree", "payu", "billdesk", "ccavenue", "instamojo", "zerodha", 
    "upstox", "groww", "angel broking", "5paisa", "sharekhan", "motilal oswal", "iifl", 
    "jm financial", "anand rathi", "avendus capital", "edelweiss", "makemytrip", 
    "goibibo", "redbus", "yatra", "oyo", "fab hotels", "zomato", "swiggy", "dunzo",
    "bigbasket", "grofers", "blinkit", "meesho", "nykaa", "purplle", "lenskart", 
    "pepperfry", "urban ladder", "firstcry", "myglamm", "mamaearth", "boat", "noise", 
    "fastrack", "myntra", "flipkart", "ola", "ola electric", "rapido", "ather energy", 
    "pure ev", "bounce", "byju's", "unacademy", "vedantu", "upgrad", "simplilearn",
    "great learning", "imarticus", "jaro education", "edureka", "intellipaat", 
    "digital vidya", "delhivery", "xpressbees", "ekart logistics", "ecom express",
    "rivigo", "blackbuck", "porter", "shadowfax", "practo", "mfine", "1mg", "tata 1mg",
    "netmeds", "pharmeasy", "medplus", "cure.fit", "urban company", "cleartax", "fisdom", 
    "smallcase", "kuvera", "policybazaar", "coverfox", "acko", "digit insurance",
    "lendingkart", "capital float", "aye finance", "stashfin", "money view", "slice",
    "navi", "freo", "simpl", "zoho", "freshworks", "chargebee", "zenoti", "whatfix", 
    "browserstack", "postman", "hasura", "appsmith", "signzy", "digio", "leegality",
    "zepto", "instamart", "milkbasket", "licious", "freshtohome", "moglix", "udaan", 
    "ofbusiness", "elasticrun",

    # USA BIG TECH / FAANG
    "google", "alphabet", "meta", "facebook", "instagram", "whatsapp", "amazon", "aws", 
    "apple", "microsoft", "azure", "netflix", "tesla", "spacex", "nvidia", "amd", "intel", 
    "qualcomm", "broadcom", "oracle", "ibm", "salesforce", "adobe", "sap", "cisco", 
    "vmware", "servicenow", "workday", "hubspot", "twilio", "okta", "crowdstrike", 
    "palo alto networks", "splunk", "datadog", "snowflake", "palantir", "openai", 
    "anthropic", "stripe", "paypal", "robinhood", "coinbase", "shopify", "walmart", 
    "target", "costco", "uber", "lyft", "fedex", "ups", "twitter", "x corp", "linkedin", 
    "pinterest", "snapchat", "tiktok", "bytedance", "reddit", "discord", "spotify", 
    "youtube", "disney", "warner bros", "comcast", "at&t", "verizon", "t-mobile", 
    "boeing", "lockheed martin", "raytheon", "general electric", "ge", "honeywell", 
    "johnson & johnson", "pfizer", "merck", "abbvie", "moderna", "unitedhealth group", 
    "jpmorgan", "goldman sachs", "morgan stanley", "bank of america", "wells fargo", 
    "citibank", "american express", "visa", "mastercard", "fidelity", "blackrock", 
    "schwab", "kpmg", "deloitte", "pwc", "ernst & young", "ey", "mckinsey", "bcg", 
    "bain", "accenture", "capgemini", "epic games", "ea", "electronic arts", "activision", 
    "blizzard", "ubisoft", "valve", "riot games", "starbucks", "mcdonald's", "pepsico", 
    "pepsi", "coca-cola", "nestle", "procter & gamble", "p&g", "unilever", "nike", 
    "adidas", "ford", "general motors", "gm", "rivian", "american airlines", "delta airlines", 
    "marriott", "hilton", "bloomberg", "gartner",

    # EUROPE & ASIA (EXTRACTED)
    "barclays", "hsbc", "lloyds", "natwest", "standard chartered", "bt group", "vodafone", 
    "bbc", "sky", "reuters", "rolls-royce", "bae systems", "astrazeneca", "gsk", 
    "tesco", "shell", "bp", "arm holdings", "sage group", "ocado", "monzo", "revolut", 
    "wise", "volkswagen", "vw", "audi", "porsche", "bmw", "mercedes-benz", "siemens", 
    "bosch", "basf", "bayer", "deutsche bank", "allianz", "ubs", "novartis", "roche", 
    "abb", "zalando", "lufthansa", "bnp paribas", "societe generale", "axa", 
    "total energies", "airbus", "lvmh", "l'oreal", "sanofi", "renault", "stellantis", 
    "orange", "carrefour", "danone", "philips", "asml", "ing group", "heineken", 
    "ikea", "h&m", "ericsson", "volvo", "novo nordisk", "maersk", "nokia", "ferrari", 
    "enel", "eni", "prada", "gucci", "alibaba", "tencent", "baidu", "xiaomi", "oppo", 
    "huawei", "lenovo", "foxconn", "tsmc", "toyota", "honda", "sony", "panasonic", 
    "hitachi", "softbank", "rakuten", "nintendo", "samsung", "lg", "sk hynix", 
    "hyundai", "kia", "grab", "gojek", "shopee", "sea limited", "dbs", "petronas", 
    "airasia", "bhp", "rio tinto", "commonwealth bank", "westpac", "anz", "nab", 
    "macquarie group", "telstra", "atlassian", "canva", "aramco", "saudi aramco", 
    "emirates", "etihad", "mtn group", "safaricom", "dangote group", "flutterwave", 
    "petrobras", "embraer", "nubank", "mercado libre", "cemex", "bimbo",
    "musc", "musc children's hospital", "one80 place", "one80 place homeless shelter",
    "optmyzr", "payben", "payben private limited", "futurense technologies", "futurense"
}

COMPANY_DB_LOWER = {c.lower().strip() for c in COMPANY_DB}

def is_company(name: str) -> bool:
    n = name.lower().strip()
    
    # STRICT BAN on Education
    edu_markers = ['university', 'college', 'school', 'institute', 'academy', 'iit', 'nit', 'bits', 'iiit', 'high school']
    if any(m in n for m in edu_markers):
        return False
        
    # SUFFIX INTELLIGENCE: If it sounds like a company, it's a company
    comp_suffixes = ['technologies', 'solutions', 'systems', 'pvt ltd', 'limited', 'inc', 'corp', 'corporation', 'llc', 'designs', 'advertising', 'pharmaceuticals', 'hospital', 'shelter', 'center']
    if any(n.endswith(s) for s in comp_suffixes):
        return True
    
    # Clean name for database check
    n_clean = re.sub(r'\b(llc|inc|ltd|corp|corporation|group|solutions|services|technologies|technology)\b', '', n).strip()
    
    if n in COMPANY_DB_LOWER or n_clean in COMPANY_DB_LOWER:
        return True
        
    for company in COMPANY_DB_LOWER:
        if len(n) > 3 and len(company) > 3:
            if company == n or (company in n and len(n) < len(company) + 5):
                return True
    return False

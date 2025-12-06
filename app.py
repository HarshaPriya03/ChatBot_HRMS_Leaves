from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import datetime
from sentence_transformers import SentenceTransformer, util
import threading
import traceback
from flask import send_from_directory
import os
import calendar

app = Flask(__name__)
CORS(app)

# --------------------- GLOBAL VARIABLES ---------------------

tokenizer = None
model = None
intent_detector = None
pagination_states = {}

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# --------------------- MODEL + DB ---------------------

def get_db():
    try:
        return mysql.connector.connect(
            host="68.178.155.255",
            user="Anika12",
            password="Anika12",
            database="ems",
            autocommit=True,
            connection_timeout=30
        )
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        raise

def ensure_connection(conn):
    try:
        conn.ping(reconnect=True, attempts=3, delay=1)
        return conn
    except mysql.connector.Error:
        try:
            conn.close()
        except:
            pass
        return get_db()

def safe_close_connection(conn):
    try:
        if conn and conn.is_connected():
            conn.close()
    except Exception as e:
        print(f"Warning: Error closing connection: {e}")

def load_model():
    print("Loading Llama 3.1 8B Instruct in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    print("✅ Llama 3.1 8B loaded in 4-bit & ready")
    return tokenizer, model

# --------------------- DATE PREPROCESSING WITH DYNAMIC HANDLING ---------------------

def preprocess_question(question: str) -> tuple:
    """
    Convert natural language dates to specific date ranges with dynamic calculation.
    """
    q_lower = question.lower()
    today = datetime.date.today()
    date_context = ""
    
    if "last month" in q_lower:
        last_month_start = (today.replace(day=1) - datetime.timedelta(days=1)).replace(day=1)
        last_month_end = today.replace(day=1) - datetime.timedelta(days=1)
        date_context = f"Date range: {last_month_start} to {last_month_end}"
        question = question.replace("last month", f"in period '{last_month_start}' to '{last_month_end}'")
    
    elif "this month" in q_lower:
        month_start = today.replace(day=1)
        month_end = datetime.date(today.year, today.month, calendar.monthrange(today.year, today.month)[1])
        date_context = f"Date range: {month_start} to {month_end}"
        question = question.replace("this month", f"in period '{month_start}' to '{month_end}'")
    
    elif "this year" in q_lower:
        year_start = today.replace(month=1, day=1)
        year_end = today.replace(month=12, day=31)
        date_context = f"Date range: {year_start} to {year_end}"
        question = question.replace("this year", f"in period '{year_start}' to '{year_end}'")
    
    elif "last week" in q_lower:
        last_week_start = today - datetime.timedelta(days=today.weekday() + 7)
        last_week_end = last_week_start + datetime.timedelta(days=6)
        date_context = f"Date range: {last_week_start} to {last_week_end}"
        question = question.replace("last week", f"in period '{last_week_start}' to '{last_week_end}'")
    
    elif "last 30 days" in q_lower or "past 30 days" in q_lower:
        date_30_days_ago = today - datetime.timedelta(days=30)
        date_context = f"Date range: {date_30_days_ago} to {today}"
        question = re.sub(r'last 30 days|past 30 days', f"in period '{date_30_days_ago}' to '{today}'", question, flags=re.IGNORECASE)
    
    # Handle "first week of Month" - FIXED TO BE DYNAMIC
    first_week_pattern = r'first week of (?:the )?(\w+)'
    first_week_match = re.search(first_week_pattern, q_lower)
    if first_week_match:
        month_name = first_week_match.group(1)
        try:
            month_num = list(calendar.month_name).index(month_name.capitalize())
            # Determine year - if month hasn't occurred yet this year, use current year, else last year
            year = today.year if month_num <= today.month else today.year - 1
            week_start = datetime.date(year, month_num, 1)
            # First week is day 1 to day 7
            week_end = datetime.date(year, month_num, min(7, calendar.monthrange(year, month_num)[1]))
            date_context = f"Date range: {week_start} to {week_end}"
            question = re.sub(
                r'first week of (?:the )?' + month_name,
                f"in period '{week_start}' to '{week_end}'",
                question,
                flags=re.IGNORECASE
            )
        except (ValueError, AttributeError):
            pass
    
    # Handle "last 7 days"
    if "last 7 days" in q_lower or "last seven days" in q_lower:
        days_ago = today - datetime.timedelta(days=7)
        date_context = f"Date range: {days_ago} to {today}"
        question = re.sub(r'last (?:7|seven) days', f"in period '{days_ago}' to '{today}'", question, flags=re.IGNORECASE)
    
    # Simple date references
    if "day before yesterday" in q_lower:
        day_before_yesterday = today - datetime.timedelta(days=2)
        date_context = f"Date: {day_before_yesterday}"
        question = question.replace("day before yesterday", f"on '{day_before_yesterday}'")
    
    elif "yesterday" in q_lower:
        yesterday = today - datetime.timedelta(days=1)
        date_context = f"Date: {yesterday}"
        question = question.replace("yesterday", f"on '{yesterday}'")
    
    elif "today" in q_lower:
        date_context = f"Date: {today}"
        question = question.replace("today", f"on '{today}'")
    
    return question, date_context

# --------------------- INTENT DETECTION ---------------------

class IntentDetector:
    def __init__(self):
        print("Loading intent detector...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.intent_examples = {
            "leavebalance": [
                "remaining leaves", "leave balance", "how many leaves left",
                "available leaves", "leaves available", "pending leave count",
                "current balance", "check balance", "balance remaining",
                "CL left", "SL available", "comp off balance", "CO balance",
                "can I take leave", "eligible for leave", "eligibility",
                "balance of", "leavebalance of", "what is balance", "latest leave balance",
                "can apply", "able to apply", "can take", "allowed to take",
                "casual leave balance", "sick leave balance", "total balance"
            ],
            "leaves": [
                "applied leaves", "leave history", "leaves taken", "took leave",
                "reason for leave", "leave from date to date",
                "when did apply", "past leaves", "leave applications",
                "how many days applied", "total days taken", "duration",
                "leaves in January", "on leave today", "who is on leave",
                "leave between dates", "applied on", "show leaves",
                "latest leave", "last leave", "recent leave", "when applied",
                "who applied", "which employee", "person applied", "members on leave",
                "phone number", "empph", "contact", "records", "list of",
                "from date", "to date", "leave period", "leave duration",
                "first leave", "when took", "leave type", "longest leave",
                "most leaves", "top employees", "how many members", "maximum",
                "sick leave", "casual leave", "comp off", "kept leave",
                "more than 3 days", "more than 5 days", "took more leaves",
                "more number of leaves", "took leaves more than", "exceeded leaves",
                "maximum leaves", "most leave days", "highest leave count",
                "employees with most leaves", "who took more", "top leave takers",
                "rejected leaves", "pending leaves", "approved leaves", "granted leaves"
            ]
        }
        
        self.intent_embeds = {
            k: self.model.encode(v)
            for k, v in self.intent_examples.items()
        }
        print("Intent detector ready")
    
    def detect(self, question: str):
        q_emb = self.model.encode(question)
        q_lower = question.lower()
        
        if any(word in q_lower for word in ['applied', 'application', 'submitted', 'apply for leave on', 'apply on']):
            return "leaves", 0.95 
        scores = {
            intent: float(util.cos_sim(q_emb, emb).max())
            for intent, emb in self.intent_embeds.items()
        }
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        
        if best_score < 0.35:
            q_lower = question.lower()
            if any(word in q_lower for word in ['balance', 'remaining', 'left', 'available', 'eligible', 'can apply', 'can take']):
                return "leavebalance", 0.6
            elif any(word in q_lower for word in ['applied', 'history', 'latest', 'reason', 'when', 'days', 'who', 'phone', 'empph', 'contact', 'from', 'to', 'duration', 'took', 'type', 'sick', 'casual', 'comp', 'kept', 'rejected', 'pending', 'approved']):
                return "leaves", 0.6
            return "unknown", best_score
        
        return best_intent, best_score

# --------------------- CONTEXT ANALYSIS WITH STATUS DETECTION ---------------------

# Replace analyze_query_context function with this improved version:

def analyze_query_context(question: str) -> dict:
    q_lower = question.lower()
    context = {
        'has_specific_email': False,
        'email': None,
        'is_general_query': False,
        'query_scope': 'unknown',
        'leave_type': None,
        'requires_aggregation': False,
        'aggregation_type': None,
        'min_days_threshold': None,
        'is_top_employees_query': False,
        'has_date_range': False,
        'status_filter': None,
        'is_application_query': False,
        'is_leave_for_date_query': False
    }
    if re.search(r'\b(when|what date|which date|on what day)\b.*\bappl(?:y|ied|ying)', q_lower):
        context['is_application_query'] = True
        context['status_filter'] = None
        context['check_applied_date'] = True
        return context
    
    if re.search(r'\bappl(?:y|ied|ying)\s+leave\s+for\s+(today|tomorrow|yesterday|\d{4}-\d{2}-\d{2}|on\s+\'\d{4}-\d{2}-\d{2}\')', q_lower):
        context['is_leave_for_date_query'] = True
        context['status_filter'] = None
        return context
    # ===== PRIORITY 1: Detect "applied" queries (MUST come first) =====
    if any(keyword in q_lower for keyword in ['applied', 'application', 'submitted', 'apply', 'applying']):
        if 'for' in q_lower:
            # Check if "for" is followed by time period (not specific date)
            if any(period in q_lower for period in ['last month', 'this month', 'last week', 'in period', 'in last', 'in this']):
                context['is_application_query'] = True
                context['status_filter'] = None
                context['check_applied_date'] = True
            # If "for" is followed by specific date, it's a leave start date query
            elif re.search(r'for\s+(today|tomorrow|yesterday|\d{4}-\d{2}-\d{2})', q_lower):
                context['is_leave_for_date_query'] = True
                context['status_filter'] = None
            else:
                # Default to application query
                context['is_application_query'] = True
                context['status_filter'] = None
                context['check_applied_date'] = True
        else:
            # No "for" keyword, definitely submission date
            context['is_application_query'] = True
            context['status_filter'] = None
            context['check_applied_date'] = True
    
    # ===== PRIORITY 2: Detect status-related queries =====
    elif any(word in q_lower for word in ['rejected', 'reject', 'denied']):
        context['status_filter'] = 2
    elif any(word in q_lower for word in ['pending', 'awaiting', 'waiting for approval']):
        context['status_filter'] = 0
    elif any(word in q_lower for word in ['approved', 'granted', 'accepted']):
        context['status_filter'] = 1
    elif any(word in q_lower for word in ['took', 'taken', 'on leave', 'kept leave', 'show leaves', 'latest leave', 'longest leave', 'most leaves']):
        context['status_filter'] = 1
    else:
        context['status_filter'] = 1

    # ===== DATE RANGE DETECTION =====
    if 'in period' in q_lower or 'between' in q_lower:
        context['has_date_range'] = True

    # ===== LEAVE TYPE DETECTION =====
    specific_leave_keywords = ['casual leave', 'sick leave', 'comp off', 'cl ', 'sl ', 'co ', 'casual', 'sick', 'comp']
    total_keywords = ['total', 'overall', 'combined', 'leave balance', 'all leaves']
    
    has_specific_type = any(keyword in q_lower for keyword in specific_leave_keywords)
    has_total_keyword = any(keyword in q_lower for keyword in total_keywords)
    
    if not has_specific_type or has_total_keyword:
        context['is_total_balance_query'] = True
    
    # ===== IMPROVED: EMAIL vs NAME DETECTION =====
    # First, try to find email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, question)
    
    if email_match:
        # Clear email found
        context['has_specific_email'] = True
        context['email'] = email_match.group()
        context['query_scope'] = 'specific_employee'
    else:
        # No email found - check if it's a name or generic query
        # Extract potential names (words that are capitalized or between quotes)
        name_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Capitalized names
            r'"([^"]+)"',  # Quoted text
            r"'([^']+)'"   # Single quoted text
        ]
        
        found_name = None
        for pattern in name_patterns:
            name_match = re.search(pattern, question)
            if name_match:
                potential_name = name_match.group(1)
                # Exclude common question words
                if potential_name.lower() not in ['who', 'what', 'when', 'where', 'which', 'how', 'can', 'show', 'give']:
                    found_name = potential_name
                    break
        
        # Check for generic employee phrases
        generic_employee_patterns = [
            r'\ba[n]?\s+employee\b',
            r'\bany\s+employee\b',
            r'\bsome\s+employee\b'
        ]
        
        is_generic = any(re.search(pat, q_lower) for pat in generic_employee_patterns)
        
        if is_generic:
            context['has_specific_email'] = False
            context['email'] = None
            context['query_scope'] = 'any_employee'
            context['is_general_query'] = True
        elif found_name:
            # Name found but no email - treat as name search
            context['has_specific_email'] = False
            context['employee_name'] = found_name
            context['query_scope'] = 'specific_employee_by_name'
        else:
            # No email, no name - check if it's a general query
            question_indicators = ['who', 'which', 'what', 'how many', 'list', 'show', 'give', 'top']
            person_plurals = ['employees', 'members', 'people', 'persons', 'staff', 'names']
            quantifiers = ['all', 'any', 'every', 'everyone', 'anyone', 'maximum', 'most']
            
            has_question_word = any(word in q_lower for word in question_indicators)
            has_plural = any(word in q_lower for word in person_plurals)
            has_quantifier = any(word in q_lower for word in quantifiers)
            
            if has_question_word or has_plural or has_quantifier:
                context['is_general_query'] = True
                context['query_scope'] = 'all_employees'
            else:
                context['query_scope'] = 'unclear'
    
    # ===== LEAVE TYPE =====
    if 'sick' in q_lower:
        context['leave_type'] = 'SICK LEAVE'
    elif 'casual' in q_lower:
        context['leave_type'] = 'CASUAL LEAVE'
    elif 'comp' in q_lower:
        context['leave_type'] = 'COMP OFF'
    
    # ===== AGGREGATION DETECTION =====
    aggregation_patterns = {
        'sum_days': [
            'more than', 'greater than', 'over', 'exceeding', 'above',
            'less than', 'below', 'under', 'fewer than',
            'most leaves', 'more leaves', 'maximum leaves', 'highest',
            'least leaves', 'minimum leaves', 'lowest', 'fewest',
            'took more', 'took most', 'took less', 'took least',
            'total days', 'sum of', 'number of days'
        ],
        'count': [
            'how many', 'count', 'number of employees', 'number of members',
            'how many employees', 'how many members', 'total employees'
        ],
        'top_ranking': [
            'top', 'top 5', 'top 10', 'first', 'highest', 'maximum',
            'most', 'best', 'leading'
        ]
    }
    
    for agg_type, keywords in aggregation_patterns.items():
        if any(keyword in q_lower for keyword in keywords):
            context['requires_aggregation'] = True
            context['aggregation_type'] = agg_type
            context['is_general_query'] = True
            context['query_scope'] = 'all_employees'
            break
    
    days_match = re.search(r'(\d+)\s*days?', q_lower)
    if days_match:
        context['min_days_threshold'] = int(days_match.group(1))
    
    top_match = re.search(r'top\s+(\d+)\s+(employees|members)', q_lower)
    if top_match:
        context['is_top_employees_query'] = True
        context['top_limit'] = int(top_match.group(1))
    
    return context

# --------------------- QUICK SYNTAX FIXES ---------------------

def quick_syntax_fix(sql: str) -> str:
    sql = re.sub(r'DATEDIFF`', r'DATEDIFF(`', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\s+NULLS\s+(FIRST|LAST)', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bILIKE\b', 'LIKE', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bltype\b', 'leavetype', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bemail\b(?![\w@])', 'empemail', sql, flags=re.IGNORECASE)
    sql = re.sub(
        r"\bempname\s*=\s*'([^']*@[^']*)'",
        r"empemail = '\1'",
        sql,
        flags=re.IGNORECASE
    )
    sql = re.sub(
        r"DATEDIFF\s*\(\s*['\"]day['\"]\s*,\s*([^,]+)\s*,\s*([^)]+)\)",
        r"DATEDIFF(\2, \1)",
        sql,
        flags=re.IGNORECASE
    )
    
    sql = re.sub(
        r"INTERVAL\s+'(\d+)\s+(day|week|month|year)s?'",
        r"INTERVAL \1 \2",
        sql,
        flags=re.IGNORECASE
    )
    
    sql = re.sub(
        r'\(?\s*CURRENT_DATE\s*-\s*INTERVAL\s+(\d+)\s+(DAY|WEEK|MONTH|YEAR)\s*\)?',
        r'DATE_SUB(CURDATE(), INTERVAL \1 \2)',
        sql,
        flags=re.IGNORECASE
    )
    
    sql = re.sub(r'\bCURRENT_DATE\(\)\b', 'CURDATE()', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bCURRENT_DATE\b', 'CURDATE()', sql, flags=re.IGNORECASE)
    
    sql = re.sub(r'\bfrom\b(?![\s`])', '`from`', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bto\b(?![\s`])', '`to`', sql, flags=re.IGNORECASE)
    
    if 'CURDATE() BETWEEN' in sql.upper():
        sql = re.sub(
            r"\s+AND\s+`?from`?\s*<=\s*['\"][\d-]+['\"]",
            "",
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r"\s+AND\s+`?to`?\s*>=\s*['\"][\d-]+['\"]",
            "",
            sql,
            flags=re.IGNORECASE
        )
    
    if not re.search(r'ORDER\s+BY\s+CAST\s*\(\s*(cl|sl|co)', sql, flags=re.IGNORECASE):
        sql = re.sub(
            r'\bORDER\s+BY\s+(cl|sl|co)\b(\s+(ASC|DESC))?',
            lambda m: f'ORDER BY CAST({m.group(1)} AS DECIMAL(10,2)){m.group(2) if m.group(2) else ""}',
            sql,
            flags=re.IGNORECASE
        )
    
    sql = re.sub(
        r"leavetype\s*=\s*'(sick|casual|comp)'",
        r"LOWER(leavetype) LIKE '%\1%'",
        sql,
        flags=re.IGNORECASE
    )
    if re.search(r'\bWHERE\b.*\bSUM\s*\(', sql, flags=re.IGNORECASE | re.DOTALL):
        pass
    if re.search(r'SELECT\s+empph\s+FROM', sql, flags=re.IGNORECASE) and \
       not re.search(r'SELECT\s+DISTINCT', sql, flags=re.IGNORECASE):
        sql = re.sub(r'SELECT\s+empph', 'SELECT DISTINCT empph', sql, flags=re.IGNORECASE)
    
    return sql

# --------------------- SCHEMA TEMPLATES WITH STATUS COLUMN ---------------------

SCHEMA_TEMPLATES = {
    "leavebalance": """
Table: leavebalance
Columns:
  - id: PRIMARY KEY
  - empname: VARCHAR (employee name)
  - empemail: VARCHAR (employee email) - CRITICAL: Use 'empemail' NOT 'email'
  - cl: VARCHAR (Casual Leave balance - CAST to DECIMAL for math)
  - sl: VARCHAR (Sick Leave balance - CAST to DECIMAL for math)
  - co: VARCHAR (Comp Off balance - CAST to DECIMAL for math)
  - lastupdate: DATETIME
  - icl, isl, ico, iupdate: initial values

RULES:
- For "can X apply for leave" → Check if balance > 0
- For "leave balance" → SELECT cl, sl, co FROM leavebalance
- Column is 'empemail' NOT 'email'
- cl/sl/co are VARCHAR → Use CAST(cl AS DECIMAL(10,2)) for comparisons
- For total balance: CAST(cl AS DECIMAL) + CAST(sl AS DECIMAL) + CAST(co AS DECIMAL)
""",
    
    "leaves": """
Table: leaves
Columns:
  - ID: PRIMARY KEY
  - empname: VARCHAR (employee name)
  - empemail: VARCHAR (employee email) - CRITICAL: Use 'empemail' NOT 'email'
  - leavetype: VARCHAR ('SICK LEAVE', 'CASUAL LEAVE', 'COMP OFF')
  - applied: DATETIME (when leave was submitted)
  - from: DATE (leave start date) - MUST use backticks `from`
  - to: DATE (leave end date) - MUST use backticks `to`
  - desg: VARCHAR (designation)
  - reason: TEXT (leave reason)
  - empph: VARCHAR (employee phone)
  - work_location: VARCHAR
  - status: INT (0=pending, 1=approved/granted, 2=rejected)
  
**CRITICAL: "APPLIED FOR" vs "APPLIED ON"**
1. "applied leave FOR [date]" → Means leave starting on that date
   - Use `from` column
   - Example: "how many applied leave FOR today" → WHERE `from` = CURDATE()

2. "applied [ON] [date]" or "who applied today" → Means submission date
   - Use `applied` column with timezone
   - Example: "who applied for leave today" → WHERE DATE(CONVERT_TZ(applied, ...)) = CURDATE()

**CRITICAL DISTINCTION: "APPLIED" vs "TOOK" QUERIES:**
"APPLIED" queries → Use 'applied' column, NO status filter
- Keywords: "applied", "application", "submitted", "submit"
- Filter: WHERE DATE(applied) = '...' (or CONVERT_TZ for timezone)
- Status: NONE (show all applications)

"TOOK" queries → Use 'from'/'to' columns, status = 1 ONLY
- Keywords: "took", "on leave", "kept leave", "took leave"
- Filter: WHERE `from` <= 'END' AND `to` >= 'START' AND status = 1
- Status: 1 (only approved)

**EXAMPLES:**
Q: "how many applied leave for today"
A: SELECT COUNT(*) FROM leaves WHERE `from` = CURDATE();
(This asks about leaves STARTING today, regardless of when applied)

Q: "how many applied leave for on '2025-12-04'"
A: SELECT COUNT(*) FROM leaves WHERE `from` = '2025-12-04';
(This asks about leaves STARTING on this date)

Q: "who applied for leave today"
A: SELECT empname, empemail FROM leaves WHERE DATE(CONVERT_TZ(applied, '+00:00', '+05:30')) = CURDATE();
(NO status filter - this asks who SUBMITTED applications today)

Q: "who applied for leave for today"
A: SELECT empname, empemail FROM leaves WHERE `from` = CURDATE();
(This asks who applied for leaves that START today)

Q: "how many members applied for leave today"
A: SELECT COUNT(DISTINCT empname) FROM leaves WHERE DATE(CONVERT_TZ(applied, '+00:00', '+05:30')) = CURDATE();
(This asks how many people SUBMITTED today)

Q: "who applied for leave in last month"
WRONG: WHERE `from` <= '2025-11-30' AND `to` >= '2025-11-01' AND status = 1
RIGHT: WHERE DATE(CONVERT_TZ(applied, '+00:00', '+05:30')) >= '2025-11-01' 
       AND DATE(CONVERT_TZ(applied, '+00:00', '+05:30')) <= '2025-11-30'

Q: "show leaves applied by John in November"
RIGHT: SELECT * FROM leaves 
       WHERE empname = 'John' 
       AND DATE(CONVERT_TZ(applied, '+00:00', '+05:30')) >= '2025-11-01' 
       AND DATE(CONVERT_TZ(applied, '+00:00', '+05:30')) <= '2025-11-30';

Q: "who took leave in last month"
RIGHT: SELECT * FROM leaves 
       WHERE `from` <= '2025-11-30' AND `to` >= '2025-11-01' AND status = 1;
       
If question contains:
- "applied" / "application" / "apply" / "applying"
- "applied for [date]" / "applied leave for [date]"

Then: DO NOT ADD "AND status = ..." to the query!

**CRITICAL STATUS FILTERING RULES:**
1. "applied"/"application"/"submitted" → NO status filter, use 'applied' column
2. "took"/"on leave"/"kept leave" → status = 1, use 'from'/'to' columns
3. "rejected"/"denied" → status = 2
4. "pending"/"awaiting approval" → status = 0
5. "approved"/"granted" → status = 1

**CRITICAL STATUS FILTERING RULES:**
1. If user asks about "applied"/"application"/"submitted" → NO status filter (show all)
2. If user asks about "took"/"on leave"/"kept leave"/"latest leave"/"show leaves" → status = 1 ONLY
3. If user asks about "rejected"/"denied" → status = 2
4. If user asks about "pending"/"awaiting approval" → status = 0
5. If user asks about "approved"/"granted" → status = 1
6. DEFAULT (when showing leave records) → status = 1

**DATE RANGE HANDLING:**

For queries with date periods, use OVERLAPPING logic:
WHERE `from` <= 'END_DATE' AND `to` >= 'START_DATE'

CALCULATING DAYS:
ALWAYS use simple DATEDIFF for aggregations:
SUM(DATEDIFF(`to`, `from`) + 1)

DO NOT USE LEAST/GREATEST - they cause calculation errors!

For simple duration (no period restriction):
DATEDIFF(`to`, `from`) + 1

**SORTING RULES:**
1. "Latest/recent applied" → ORDER BY applied DESC
2. "Latest/recent leave" (no "applied") → ORDER BY `from` DESC  
3. "Who took more/most" → ORDER BY total_days DESC (or leave_count DESC)
4. "First week" / "first leave" → ORDER BY `from` ASC

**GROUPING RULES:**
1. ALWAYS use empname for GROUP BY (NOT empemail)
2. Use COUNT(DISTINCT empname) for counting unique employees
3. empname is the primary identifier, empemail can be NULL or duplicate

EXAMPLES:

Q: "who applied for leave today"
A: SELECT empname, empemail 
   FROM leaves 
   WHERE DATE(CONVERT_TZ(applied, '+00:00', '+05:30')) = CURDATE();
(NO status filter!)

Q: "how many applied leave for today"
A: SELECT COUNT(*) 
   FROM leaves 
   WHERE DATE(CONVERT_TZ(from, '+00:00', '+05:30')) = CURDATE();

Q: "who applied for leave for today"
A: SELECT empname, empemail, `from`, `to`, leavetype 
   FROM leaves 
   WHERE DATE(CONVERT_TZ(from, '+00:00', '+05:30')) = CURDATE();
(NO status filter!)

Q: "how many members applied for leave today"
A: SELECT COUNT(DISTINCT empname) 
   FROM leaves 
   WHERE DATE(CONVERT_TZ(from, '+00:00', '+05:30')) = CURDATE();
(NO status filter!)

Q: "list all applications submitted yesterday"
A: SELECT empname, empemail, `from`, `to`, leavetype, reason
   FROM leaves
   WHERE DATE(CONVERT_TZ(applied, '+00:00', '+05:30')) = DATE_SUB(CURDATE(), INTERVAL 1 DAY);
(NO status filter!)

Q: "who is on leave today"
A: SELECT empname, empemail FROM leaves WHERE CURDATE() BETWEEN `from` AND `to` AND status = 1;
(status = 1 because "on leave" means approved)

Q: "show rejected leaves"
A: SELECT empname, empemail, `from`, `to`, reason FROM leaves WHERE status = 2;

Q: "employees who took more than 5 days in October"
A: SELECT empname, empemail, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days
   FROM leaves 
   WHERE `from` <= '2025-10-31' AND `to` >= '2025-10-01' AND status = 1
   GROUP BY empname, empemail 
   HAVING total_days > 5;
(Overlapping filter + simple SUM with DATEDIFF)
"""
}

def get_schema_for_intent(intent: str) -> str:
    if intent == "leavebalance":
        return SCHEMA_TEMPLATES["leavebalance"]
    elif intent == "leaves":
        return SCHEMA_TEMPLATES["leaves"]
    else:
        return SCHEMA_TEMPLATES["leavebalance"] + "\n" + SCHEMA_TEMPLATES["leaves"]

# --------------------- LLAMA PROMPT BUILDING WITH STATUS AWARENESS ---------------------

def build_llama_prompt(user_question: str, intent: str, context: dict) -> str:
    schema = get_schema_for_intent(intent)
    print("="*80)
    print("DEBUG - CONTEXT ANALYSIS")
    print("="*80)
    print(f"Question: {user_question}")
    print(f"Intent: {intent}")
    print(f"is_application_query: {context.get('is_application_query')}")
    print(f"check_applied_date: {context.get('check_applied_date')}")
    print(f"status_filter: {context.get('status_filter')}")
    print("="*80)
    
    # Build status filter hint
    status_hint = ""
    if context.get('is_leave_for_date_query'):
        status_hint = """CRITICAL: This query asks about leaves FOR a specific date (leave start date).
                        Use `from` column, NOT 'applied' column.
                        Example: WHERE `from` = '2025-12-04'
                        No status filter needed unless specified."""
    elif context.get('is_application_query'):
        status_hint = """CRITICAL: This query asks about application SUBMISSION date.
                        Use 'applied' column with timezone conversion.
                        Example: WHERE DATE(CONVERT_TZ(applied, '+00:00', '+05:30')) = CURDATE()
                        NO status filter - show all applications."""
    elif context.get('status_filter') is not None:
        status_hint = f"CRITICAL: Add 'AND status = {context['status_filter']}' to WHERE clause"
    # elif context.get('is_application_query'):
    #     status_hint = "CRITICAL: DO NOT ADD STATUS FILTER - This is an application query"
    elif intent == "leaves":
        status_hint = "CRITICAL: Add 'AND status = 1' to WHERE clause (only approved leaves)"
    
    context_hint = ""
    if context['has_specific_email']:
        context_hint = f"FILTER: WHERE empemail = '{context['email']}'"
    elif context['is_general_query']:
        context_hint = "FILTER: Query ALL employees. Include empname and empemail in SELECT."
    
    q_lower = user_question.lower()
    intent_hint = ""
    
    if intent == "leavebalance":
        if context.get('is_total_balance_query', False):
            if any(word in q_lower for word in ['less than', '<', 'below', 'under']):
                intent_hint = """TASK: Query 'leavebalance' table for TOTAL leave balance.
Calculate: (CAST(cl AS DECIMAL(10,2)) + CAST(sl AS DECIMAL(10,2)) + CAST(co AS DECIMAL(10,2))) AS total_balance
Include empname, empemail, cl, sl, co, total_balance in SELECT."""
            elif any(word in q_lower for word in ['greater than', '>', 'above', 'more than']):
                intent_hint = "TASK: Query 'leavebalance' table for TOTAL balance > threshold."
        elif context['leave_type']:
            leave_col = 'cl' if 'CASUAL' in context['leave_type'] else 'sl' if 'SICK' in context['leave_type'] else 'co'
            intent_hint = f"TASK: Query 'leavebalance' table. Filter by {leave_col}. Use CAST({leave_col} AS DECIMAL(10,2))."
        elif "can" in q_lower and "apply" in q_lower:
            intent_hint = "TASK: Check if employee can apply for leave by verifying balance > 0"
        else:
            intent_hint = "TASK: Query 'leavebalance' table. Return cl, sl, co values."
    
    elif intent == "leaves":
        if context.get('is_application_query') and context.get('check_applied_date'):
        # This is an "applied" query - check 'applied' column, no status filter
            if 'in period' in q_lower or context.get('has_date_range'):
                intent_hint = f"""TASK: Find employees who APPLIED for leave in period.
CRITICAL: Use 'applied' column, NOT 'from'/'to' columns
Query: SELECT empname, empemail, `from`, `to`, leavetype FROM leaves
WHERE DATE(CONVERT_TZ(applied, '+00:00', '+05:30')) >= 'START_DATE' 
  AND DATE(CONVERT_TZ(applied, '+00:00', '+05:30')) <= 'END_DATE'
NO STATUS FILTER - show all applications regardless of status"""
            else:
                intent_hint = f"""TASK: Find employees who APPLIED for leave on specific date.
CRITICAL: Use 'applied' column, NOT 'from'/'to' columns
Query: SELECT empname, empemail, `from`, `to`, leavetype FROM leaves
WHERE DATE(CONVERT_TZ(applied, '+00:00', '+05:30')) = 'TARGET_DATE'
NO STATUS FILTER - show all applications regardless of status"""
        if context.get('requires_aggregation', False):
            days_threshold = context.get('min_days_threshold')
            top_match = re.search(r'top\s*(\d+)', q_lower)
            limit_count = top_match.group(1) if top_match else '5'
            agg_type = context.get('aggregation_type', 'sum_days')
            
            has_period = context.get('has_date_range', False) or any(period in q_lower for period in ['this month', 'last month', 'this year', 'last week', 'in period'])
            
            # Extract period dates if present (these are injected by preprocess_question)
            period_match = re.search(r"in period '([\d-]+)' to '([\d-]+)'", user_question)
            period_start = period_match.group(1) if period_match else None
            period_end = period_match.group(2) if period_match else None
            
            # Build the CASE statement for period-aware queries
            if has_period and period_start and period_end:
                case_statement = f"""CASE 
       WHEN `from` >= '{period_start}' AND `to` <= '{period_end}' THEN DATEDIFF(`to`, `from`) + 1
       WHEN `from` < '{period_start}' AND `to` <= '{period_end}' THEN DATEDIFF(`to`, '{period_start}') + 1
       WHEN `from` >= '{period_start}' AND `to` > '{period_end}' THEN DATEDIFF('{period_end}', `from`) + 1
       WHEN `from` < '{period_start}' AND `to` > '{period_end}' THEN DATEDIFF('{period_end}', '{period_start}') + 1
       ELSE 0
     END"""
            else:
                # No period - simple DATEDIFF
                case_statement = "DATEDIFF(`to`, `from`) + 1"
            
            if agg_type == 'count':
                if days_threshold:
                    if has_period:
                        intent_hint = f"""TASK: COUNT employees who took more than {days_threshold} days in given period.
Use overlapping: WHERE `from` <= '{period_end}' AND `to` >= '{period_start}' AND status = 1
Calculate: SUM({case_statement}) AS total_days
Structure: SELECT COUNT(*) FROM (
  SELECT empemail 
  FROM leaves 
  WHERE `from` <= '{period_end}' AND `to` >= '{period_start}' AND status = 1
  GROUP BY empemail 
  HAVING SUM({case_statement}) > {days_threshold}
) AS subquery;"""
                    else:
                        intent_hint = f"""TASK: COUNT employees who took more than {days_threshold} total days.
Structure: SELECT COUNT(*) FROM (
  SELECT empemail 
  FROM leaves 
  WHERE status = 1 
  GROUP BY empemail 
  HAVING SUM(DATEDIFF(`to`, `from`) + 1) > {days_threshold}
) AS subquery;"""
                else:
                    intent_hint = f"TASK: COUNT distinct employees. {status_hint}. Use COUNT(DISTINCT empname)"
            
            elif agg_type == 'top_ranking' or 'top' in q_lower or 'most' in q_lower:
                if has_period:
                    intent_hint = f"""TASK: Find top {limit_count} employees with most leave days in period.
Use overlapping: WHERE `from` <= '{period_end}' AND `to` >= '{period_start}' AND status = 1
Calculate: SUM({case_statement}) AS total_days
Include: empname, empemail, COUNT(*) AS leave_count, total_days
GROUP BY empname, empemail
ORDER BY total_days DESC
LIMIT {limit_count}"""
                else:
                    intent_hint = f"""TASK: Find top {limit_count} employees with most total leave days.
{status_hint}
Calculate: SUM(DATEDIFF(`to`, `from`) + 1) AS total_days
Include: empname, empemail, COUNT(*) AS leave_count, total_days
GROUP BY empname, empemail
ORDER BY total_days DESC
LIMIT {limit_count}"""
            
            else:
                # Comparison queries (more than, less than, etc.)
                comparison_operators = {
                    'more than': '>', 'greater than': '>', 'over': '>', 'above': '>',
                    'less than': '<', 'below': '<', 'under': '<', 'fewer than': '<'
                }
                
                operator = '>'
                for phrase, op in comparison_operators.items():
                    if phrase in q_lower:
                        operator = op
                        break
                
                if days_threshold:
                    if has_period:
                        intent_hint = f"""TASK: Find employees who took {operator} {days_threshold} days in given period.
Use overlapping: WHERE `from` <= '{period_end}' AND `to` >= '{period_start}' AND status = 1
Calculate: SUM({case_statement}) AS total_days
Include: empname, empemail, total_days
GROUP BY empname, empemail
HAVING total_days {operator} {days_threshold}
ORDER BY total_days DESC"""
                    else:
                        intent_hint = f"""TASK: Find employees who took {operator} {days_threshold} total days.
{status_hint}
Calculate: SUM(DATEDIFF(`to`, `from`) + 1) AS total_days
Include: empname, empemail, total_days
GROUP BY empname, empemail
HAVING total_days {operator} {days_threshold}
ORDER BY total_days DESC"""
                else:
                    if has_period:
                        intent_hint = f"""TASK: Find employees with most leave days in period.
Use overlapping: WHERE `from` <= '{period_end}' AND `to` >= '{period_start}' AND status = 1
Calculate: SUM({case_statement}) AS total_days
Include: empname, empemail, COUNT(*) AS leave_count, total_days
GROUP BY empname, empemail
ORDER BY total_days DESC"""
                    else:
                        intent_hint = f"""TASK: Find employees with most total leave days.
{status_hint}
Calculate: SUM(DATEDIFF(`to`, `from`) + 1) AS total_days
Include: empname, empemail, COUNT(*) AS leave_count, total_days
GROUP BY empname, empemail
ORDER BY total_days DESC"""
        
        elif any(word in q_lower for word in ['latest', 'recent', 'last']) and 'last month' not in q_lower and 'last week' not in q_lower:
            leave_type_filter = ""
            if context['leave_type']:
                leave_type_filter = f" AND LOWER(leavetype) LIKE '%{context['leave_type'].split()[0].lower()}%'"
            
            if context['has_specific_email']:
                intent_hint = f"TASK: Find most recent leave{leave_type_filter} for {context['email']}. {status_hint}. ORDER BY `from` DESC LIMIT 1"
            else:
                intent_hint = f"TASK: Find most recent leave{leave_type_filter}. {status_hint}. ORDER BY `from` DESC LIMIT 1. Show empname, empemail, `from`, `to`, leavetype"
        
        elif any(word in q_lower for word in ['duration', 'how many days', 'how long']):
            intent_hint = f"TASK: Calculate duration using DATEDIFF(`to`, `from`) + 1. {status_hint}"
        
        elif "phone" in q_lower or "contact" in q_lower:
            intent_hint = f"TASK: SELECT DISTINCT empph from leaves. {status_hint if not context.get('is_application_query') else ''} LIMIT 1"
        
        elif any(word in q_lower for word in ['on leave', 'who is on leave', 'members on leave', 'kept leave']):
            date_pattern = r"on\s+'([\d-]+)'"
            date_match = re.search(date_pattern, user_question)
            if date_match:
                specific_date = date_match.group(1)
                intent_hint = f"TASK: Find employees on leave on {specific_date}. WHERE '{specific_date}' BETWEEN `from` AND `to` AND status = 1"
            elif 'in period' in q_lower:
                intent_hint = f"TASK: Find employees with leaves in period. Use overlapping: WHERE `from` <= 'END_DATE' AND `to` >= 'START_DATE' AND status = 1"
            else:
                intent_hint = "TASK: Find employees on leave today. WHERE CURDATE() BETWEEN `from` AND `to` AND status = 1"
        
        elif any(word in q_lower for word in ['longest', 'maximum duration', 'max duration']):
            if context.get('has_date_range', False):
                intent_hint = f"TASK: Find longest leave in period. Calculate: DATEDIFF(`to`, `from`) + 1 AS duration. Use overlapping WHERE. {status_hint}. ORDER BY duration DESC LIMIT 1"
            else:
                intent_hint = f"TASK: Find longest leave overall. Calculate: DATEDIFF(`to`, `from`) + 1 AS duration. {status_hint}. ORDER BY duration DESC LIMIT 1"

    warning = ""
    if context['is_general_query'] or not context['has_specific_email']:
        warning = "\n⚠️ CRITICAL: Query should work for ALL employees without email filtering.\n"

    system_prompt = f"""You are a MySQL query generator for an Employee Management System.

DATABASE SCHEMA:
{schema}

MYSQL SYNTAX RULES:
- Tables: 'leavebalance', 'leaves'
- Columns: 'empemail' (NOT 'email'), 'leavetype' (NOT 'ltype')
- Reserved words: from, to (use backticks `from`, `to`)
- Use CURDATE() not CURRENT_DATE
- Date format: 'YYYY-MM-DD'
- DATEDIFF(end, start) for MySQL
- Status column: 0=pending, 1=approved, 2=rejected

**CRITICAL RULES FOR "A/AN EMPLOYEE" QUERIES:**
- "a employee", "an employee", "any employee" → NO WHERE clause on empemail
- "the employee", "specific employee", or actual email → USE WHERE empemail = '...'
- Generic queries like "latest leave", "recent application" → NO email filter
- Never use placeholder values like 'employee_email' or 'user@example.com' in WHERE clauses


{status_hint}
{context_hint}
{intent_hint}
{warning}

# Replace the entire examples section in build_llama_prompt with this:

ESSENTIAL EXAMPLES:

✓ "APPLIED" QUERIES (NO status filter, use timezone conversion)
✓ "TOOK" QUERIES (status = 1)
✓ PERIOD-AWARE: Use CASE statement for accurate day calculation within period

# ==================== APPLICATION QUERIES (NO STATUS FILTER) ====================

Q: "for which date maximum members applied for leave"
A: SELECT DATE(CONVERT_TZ(applied, '+00:00', '+05:30')) AS application_date, COUNT(*) AS total_applications
   FROM leaves
   GROUP BY DATE(CONVERT_TZ(applied, '+00:00', '+05:30'))
   ORDER BY total_applications DESC
   LIMIT 1;

Q: "on which day did most employees apply for leave"
A: SELECT DATE(CONVERT_TZ(applied, '+00:00', '+05:30')) AS date, COUNT(DISTINCT empname) AS members_applied
   FROM leaves
   GROUP BY DATE(CONVERT_TZ(applied, '+00:00', '+05:30'))
   ORDER BY members_applied DESC
   LIMIT 1;
   
Q: "who applied for leave today"
A: SELECT empname, empemail FROM leaves WHERE DATE(CONVERT_TZ(applied, '+00:00', '+05:30')) = CURDATE();

Q: "what is the recent leave applied by an employee"
A: SELECT empname, empemail, `from`, `to`, leavetype, CONVERT_TZ(applied, '+00:00', '+05:30') as applied_time 
   FROM leaves 
   ORDER BY applied DESC 
   LIMIT 1;

Q: "what is the latest leave applied by a employee"
A: SELECT empname, empemail, `from`, `to`, leavetype, CONVERT_TZ(applied, '+00:00', '+05:30') as applied_time 
   FROM leaves 
   ORDER BY applied DESC 
   LIMIT 1;

Q: "show me the most recent leave application"
A: SELECT empname, empemail, `from`, `to`, leavetype, CONVERT_TZ(applied, '+00:00', '+05:30') as applied_time 
   FROM leaves 
   ORDER BY applied DESC 
   LIMIT 1;

Q: "what is the latest leave"
A: SELECT empname, empemail, `from`, `to`, leavetype, reason 
   FROM leaves 
   WHERE status = 1 
   ORDER BY `from` DESC 
   LIMIT 1;

Q: "who took the most recent leave"
A: SELECT empname, empemail, `from`, `to`, leavetype 
   FROM leaves 
   WHERE status = 1 
   ORDER BY `from` DESC 
   LIMIT 1;

# ==================== PERIOD-AWARE QUERIES (WITH CASE STATEMENT) ====================

Q: "give me all leave records of john@example.com where he took leaves for more than 1 day this year"
A: SELECT empname, `from`, `to`, leavetype, reason,
  (
    CASE 
      WHEN `from` >= '2025-01-01' AND `to` <= '2025-12-31' THEN DATEDIFF(`to`, `from`) + 1
      WHEN `from` <  '2025-01-01' AND `to` <= '2025-12-31' THEN DATEDIFF(`to`, '2025-01-01') + 1
      WHEN `from` >= '2025-01-01' AND `to` >  '2025-12-31' THEN DATEDIFF('2025-12-31', `from`) + 1
      WHEN `from` <  '2025-01-01' AND `to` >  '2025-12-31' THEN DATEDIFF('2025-12-31', '2025-01-01') + 1
      ELSE 0
    END
  ) AS days
FROM leaves
WHERE empemail = 'john@example.com'
  AND `from` <= '2025-12-31'
  AND `to` >= '2025-01-01'
  AND status = 1
HAVING days > 1
ORDER BY `from` DESC;


Q: "who took more leaves in this month"
A: SELECT empname, 
   COUNT(*) AS leave_count,
   SUM(
     CASE 
       WHEN `from` >= '2025-12-01' AND `to` <= '2025-12-31' THEN DATEDIFF(`to`, `from`) + 1
       WHEN `from` < '2025-12-01' AND `to` <= '2025-12-31' THEN DATEDIFF(`to`, '2025-12-01') + 1
       WHEN `from` >= '2025-12-01' AND `to` > '2025-12-31' THEN DATEDIFF('2025-12-31', `from`) + 1
       WHEN `from` < '2025-12-01' AND `to` > '2025-12-31' THEN DATEDIFF('2025-12-31', '2025-12-01') + 1
       ELSE 0
     END
   ) AS total_days
   FROM leaves 
   WHERE `from` <= '2025-12-31' AND `to` >= '2025-12-01' AND status = 1 
   GROUP BY empname 
   ORDER BY total_days DESC;

Q: "for how many days did gorapallimeghashyam@gmail.com take leave in this month"
A: SELECT SUM(
     CASE 
       WHEN `from` >= '2025-12-01' AND `to` <= '2025-12-31' THEN DATEDIFF(`to`, `from`) + 1
       WHEN `from` < '2025-12-01' AND `to` <= '2025-12-31' THEN DATEDIFF(`to`, '2025-12-01') + 1
       WHEN `from` >= '2025-12-01' AND `to` > '2025-12-31' THEN DATEDIFF('2025-12-31', `from`) + 1
       WHEN `from` < '2025-12-01' AND `to` > '2025-12-31' THEN DATEDIFF('2025-12-31', '2025-12-01') + 1
       ELSE 0
     END
   ) AS total_days
   FROM leaves 
   WHERE `from` <= '2025-12-31' AND `to` >= '2025-12-01' AND status = 1 AND empemail = 'gorapallimeghashyam@gmail.com';

Q: "employees who took more than 3 days this month"
A: SELECT empname, SUM(
     CASE 
       WHEN `from` >= '2025-12-01' AND `to` <= '2025-12-31' THEN DATEDIFF(`to`, `from`) + 1
       WHEN `from` < '2025-12-01' AND `to` <= '2025-12-31' THEN DATEDIFF(`to`, '2025-12-01') + 1
       WHEN `from` >= '2025-12-01' AND `to` > '2025-12-31' THEN DATEDIFF('2025-12-31', `from`) + 1
       WHEN `from` < '2025-12-01' AND `to` > '2025-12-31' THEN DATEDIFF('2025-12-31', '2025-12-01') + 1
       ELSE 0
     END
   ) AS total_days
   FROM leaves
   WHERE `from` <= '2025-12-31' AND `to` >= '2025-12-01' AND status = 1
   GROUP BY empname
   HAVING total_days > 3
   ORDER BY total_days DESC;

Q: "top 5 employees who took most leaves in period '2025-10-01' to '2025-10-31'"
A: SELECT empname, empemail, COUNT(*) AS leave_count, SUM(
     CASE 
       WHEN `from` >= '2025-10-01' AND `to` <= '2025-10-31' THEN DATEDIFF(`to`, `from`) + 1
       WHEN `from` < '2025-10-01' AND `to` <= '2025-10-31' THEN DATEDIFF(`to`, '2025-10-01') + 1
       WHEN `from` >= '2025-10-01' AND `to` > '2025-10-31' THEN DATEDIFF('2025-10-31', `from`) + 1
       WHEN `from` < '2025-10-01' AND `to` > '2025-10-31' THEN DATEDIFF('2025-10-31', '2025-10-01') + 1
       ELSE 0
     END
   ) AS total_days 
   FROM leaves WHERE `from` <= '2025-10-31' AND `to` >= '2025-10-01' AND status = 1 
   GROUP BY empname, empemail ORDER BY total_days DESC LIMIT 5;

Q: "employees who took more than 5 days in period '2025-10-01' to '2025-10-31'"
A: SELECT empname, empemail, SUM(
     CASE 
       WHEN `from` >= '2025-10-01' AND `to` <= '2025-10-31' THEN DATEDIFF(`to`, `from`) + 1
       WHEN `from` < '2025-10-01' AND `to` <= '2025-10-31' THEN DATEDIFF(`to`, '2025-10-01') + 1
       WHEN `from` >= '2025-10-01' AND `to` > '2025-10-31' THEN DATEDIFF('2025-10-31', `from`) + 1
       WHEN `from` < '2025-10-01' AND `to` > '2025-10-31' THEN DATEDIFF('2025-10-31', '2025-10-01') + 1
       ELSE 0
     END
   ) AS total_days 
   FROM leaves WHERE `from` <= '2025-10-31' AND `to` >= '2025-10-01' AND status = 1 
   GROUP BY empname, empemail HAVING total_days > 5 ORDER BY total_days DESC;

Q: "how many members took more than 3 days in period '2025-11-01' to '2025-11-30'"
A: SELECT COUNT(*) FROM (
   SELECT empemail FROM leaves 
   WHERE `from` <= '2025-11-30' AND `to` >= '2025-11-01' AND status = 1 
   GROUP BY empemail 
   HAVING SUM(
     CASE 
       WHEN `from` >= '2025-11-01' AND `to` <= '2025-11-30' THEN DATEDIFF(`to`, `from`) + 1
       WHEN `from` < '2025-11-01' AND `to` <= '2025-11-30' THEN DATEDIFF(`to`, '2025-11-01') + 1
       WHEN `from` >= '2025-11-01' AND `to` > '2025-11-30' THEN DATEDIFF('2025-11-30', `from`) + 1
       WHEN `from` < '2025-11-01' AND `to` > '2025-11-30' THEN DATEDIFF('2025-11-30', '2025-11-01') + 1
       ELSE 0
     END
   ) > 3
) AS subquery;

Q: "top 5 employees who took most leaves this year"
A: SELECT empname, COUNT(*) AS leave_count, SUM(
     CASE 
       WHEN `from` >= '2025-01-01' AND `to` <= '2025-12-31' THEN DATEDIFF(`to`, `from`) + 1
       WHEN `from` < '2025-01-01' AND `to` <= '2025-12-31' THEN DATEDIFF(`to`, '2025-01-01') + 1
       WHEN `from` >= '2025-01-01' AND `to` > '2025-12-31' THEN DATEDIFF('2025-12-31', `from`) + 1
       WHEN `from` < '2025-01-01' AND `to` > '2025-12-31' THEN DATEDIFF('2025-12-31', '2025-01-01') + 1
       ELSE 0
     END
   ) AS total_days 
   FROM leaves 
   WHERE `from` <= '2025-12-31' AND `to` >= '2025-01-01' AND status = 1 
   GROUP BY empname 
   ORDER BY total_days DESC 
   LIMIT 5;

# ==================== SIMPLE QUERIES (NO PERIOD FILTER) ====================

Q: "who kept leave in period '2025-11-01' to '2025-11-07'"
A: SELECT empname, empemail, `from`, `to`, leavetype FROM leaves WHERE `from` <= '2025-11-07' AND `to` >= '2025-11-01' AND status = 1;

Q: "give me all leave records of employees who took leave in last 10 days"
A: SELECT empname, `from`, `to`, leavetype, reason 
   FROM leaves 
   WHERE `from` <= CURDATE() AND `to` >= DATE_SUB(CURDATE(), INTERVAL 10 DAY) AND status = 1 
   ORDER BY `from` DESC;

Q: "for how many days did gorapallimeghashyam@gmail.com take his last leave"
A: SELECT DATEDIFF(`to`, `from`) + 1 AS duration 
   FROM leaves 
   WHERE empemail = 'gorapallimeghashyam@gmail.com' AND status = 1 
   ORDER BY `from` DESC 
   LIMIT 1;

Q: "latest leave of john@example.com"
A: SELECT empname, `from`, `to`, leavetype, reason FROM leaves WHERE empemail = 'john@example.com' AND status = 1 ORDER BY `from` DESC LIMIT 1;

Q: "who is on leave today"
A: SELECT empname, empemail, `from`, `to`, leavetype FROM leaves WHERE CURDATE() BETWEEN `from` AND `to` AND status = 1;

Q: "who took longest duration leave this year"
A: SELECT empname, empemail, DATEDIFF(`to`, `from`) + 1 AS duration FROM leaves WHERE YEAR(`from`) = YEAR(CURDATE()) AND status = 1 ORDER BY duration DESC LIMIT 1;

Q: "top 5 employees who took most leaves"
A: SELECT empname, empemail, COUNT(*) AS leave_count, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days FROM leaves WHERE status = 1 GROUP BY empname, empemail ORDER BY total_days DESC LIMIT 5;

Q: "phone number of john@example.com"
A: SELECT DISTINCT empph FROM leaves WHERE empemail = 'john@example.com' LIMIT 1;

# ==================== LEAVE BALANCE QUERIES ====================

Q: "leave balance of john@example.com"
A: SELECT cl, sl, co FROM leavebalance WHERE empemail = 'john@example.com';

Q: "who has lowest casual leave balance"
A: SELECT empname, empemail, cl FROM leavebalance ORDER BY CAST(cl AS DECIMAL(10,2)) ASC LIMIT 1;

Q: "list employees with negative leave balance"
A: SELECT empname, cl, sl, co, 
   (CAST(cl AS DECIMAL(10,2)) + CAST(sl AS DECIMAL(10,2)) + CAST(co AS DECIMAL(10,2))) AS total_balance 
   FROM leavebalance 
   WHERE (CAST(cl AS DECIMAL(10,2)) + CAST(sl AS DECIMAL(10,2)) + CAST(co AS DECIMAL(10,2))) < 0;

Q: "can john@example.com apply for casual leave"
A: SELECT cl FROM leavebalance WHERE empemail = 'john@example.com' AND CAST(cl AS DECIMAL(10,2)) > 0;
Now generate MySQL query for:"""
    
    print("="*80)
    print("DEBUG - FINAL PROMPT SENT TO LLM")
    print("="*80)
    print("STATUS HINT:")
    print(repr(status_hint))  # repr shows if it's empty string
    print("\nFULL SYSTEM PROMPT:")
    print(system_prompt[:2000])  # First 2000 chars
    print("="*80)
    

    user_message = f"Question: {user_question}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

# --------------------- SQL GENERATION ---------------------

def generate_sql(tokenizer, model, messages: list, max_new_tokens: int = 400) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print("="*80)
    print("DEBUG - LLM INPUT")
    print("="*80)
    print(prompt[-1500:])  # Last 1500 chars of prompt
    print("="*80)
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_ids = out[0][inputs['input_ids'].shape[1]:]
    fixed_sql = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    fixed_sql = re.sub(r'```sql\s*', '', fixed_sql, flags=re.IGNORECASE | re.DOTALL)
    fixed_sql = re.sub(r'```\s*', '', fixed_sql, flags=re.IGNORECASE | re.DOTALL)
    fixed_sql = fixed_sql.strip()

    if not fixed_sql.endswith(';'):
        fixed_sql = fixed_sql.rstrip() + ';'

    fixed_sql = quick_syntax_fix(fixed_sql)
    print("="*80)
    print("DEBUG - LLM OUTPUT")
    print("="*80)
    print(f"Generated SQL: {fixed_sql}")
    print("="*80)
    return fixed_sql

# --------------------- LLM REPAIR ---------------------

def repair_sql_with_llm(sql: str, error_msg: str, tokenizer, model, schema: str, original_question: str, intent: str) -> str:
    sql = quick_syntax_fix(sql)
    error_lower = error_msg.lower()

    if "'ltype'" in error_msg or "ltype" in error_lower:
        repair_hint = "ERROR: Column 'ltype' does not exist. Use 'leavetype' instead."
        sql = re.sub(r'\bltype\b', 'leavetype', sql, flags=re.IGNORECASE)
    elif "'email'" in error_msg or (("column" in error_lower or "unknown" in error_lower) and "email" in error_lower):
        repair_hint = "ERROR: Column 'email' doesn't exist. Use 'empemail' instead."
        sql = re.sub(r'\bemail\b(?![\w@])', 'empemail', sql, flags=re.IGNORECASE)
    elif "datediff" in error_lower:
        repair_hint = "ERROR: DATEDIFF in MySQL takes 2 params: DATEDIFF(end_date, start_date)"
    elif "reserved" in error_lower or "syntax" in error_lower:
        repair_hint = "ERROR: 'from' and 'to' are reserved keywords. Use backticks: `from`, `to`"
        sql = re.sub(r'\bfrom\b(?![`\s])', '`from`', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bto\b(?![`\s])', '`to`', sql, flags=re.IGNORECASE)
    else:
        repair_hint = f"ERROR: {error_msg[:200]}"

    system_prompt = f"""You are a MySQL query debugger. Fix the broken SQL query.
{repair_hint}
DATABASE SCHEMA:
{schema}
ORIGINAL QUESTION: {original_question}
BROKEN SQL:
{sql}
Provide ONLY the corrected SQL query, nothing else."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Fix the SQL:"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_ids = out[0][inputs['input_ids'].shape[1]:]
    fixed_sql = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    fixed_sql = re.sub(r'```sql\s*', '', fixed_sql, flags=re.IGNORECASE | re.DOTALL)
    fixed_sql = re.sub(r'```\s*', '', fixed_sql, flags=re.IGNORECASE | re.DOTALL)
    fixed_sql = fixed_sql.strip()

    if not fixed_sql.endswith(';'):
        fixed_sql = fixed_sql.rstrip() + ';'

    fixed_sql = quick_syntax_fix(fixed_sql)

    return fixed_sql

# --------------------- VALIDATION & EXECUTION ---------------------

def validate_and_repair_sql(conn, sql: str, tokenizer, model, schema: str, original_question: str, intent: str, max_attempts: int = 3) -> tuple:
    current_sql = sql
    for attempt in range(max_attempts):
        cursor = None
        try:
            conn.ping(reconnect=True, attempts=2, delay=0.5)
            cursor = conn.cursor()
            sql_to_validate = current_sql.rstrip(';').strip()
            cursor.execute(f"EXPLAIN {sql_to_validate}")
            cursor.fetchall()
            return current_sql, True, attempt + 1
        except mysql.connector.Error as e:
            error_msg = str(e)
            
            if attempt < max_attempts - 1:
                current_sql = repair_sql_with_llm(current_sql, error_msg, tokenizer, model, schema, original_question, intent)
            else:
                return current_sql, False, attempt + 1
        finally:
            if cursor:
                cursor.close()

    return current_sql, False, max_attempts

def run_query(conn, sql: str):
    cursor = None
    try:
        conn.ping(reconnect=True, attempts=2, delay=0.5)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description] if cursor.description else []
        return cols, rows
    except mysql.connector.Error as e:
        raise Exception(f"Query failed: {e}")
    finally:
        if cursor:
            cursor.close()

# --------------------- VERBALIZATION ---------------------
def clean_null_values(rows, cols):
    """Convert NULL values to 0 for numeric columns."""
    cleaned_rows = []
    for row in rows:
        cleaned_row = []
        for i, val in enumerate(row):
            col_name = cols[i].lower()
            # If column name suggests numeric value and value is None, replace with 0
            if val is None and any(keyword in col_name for keyword in ['days', 'count', 'total', 'balance', 'cl', 'sl', 'co', 'duration']):
                cleaned_row.append(0)
            else:
                cleaned_row.append(val)
        cleaned_rows.append(tuple(cleaned_row))
    return cleaned_rows


def verbalize_results(tokenizer, model, user_question: str, cols: list, rows: list, intent: str, page_size: int = 3, page_num: int = 1) -> str:
    if not rows:
        return "No results found for your query."
    
    # Handle single value responses (counts, phone numbers, etc.)
    if len(cols) == 1 and len(rows) == 1:
        col_name = cols[0]
        value = rows[0][0]
        
        if 'COUNT' in col_name.upper():
            q_lower = user_question.lower()
            if 'how many' in q_lower or 'count' in q_lower:
                entity = 'employees'
                if 'leaves' in q_lower or 'applications' in q_lower:
                    entity = 'leave applications'
                return f"{value} {entity} match your criteria."
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, user_question)
        
        if col_name.lower() in ['empph', 'phone', 'contact']:
            if email_match:
                return f"The phone number for {email_match.group()} is: {value}"
            else:
                return f"The phone number is: {value}"
        elif col_name.lower() in ['cl', 'sl', 'co']:
            return f"The {col_name.upper()} balance is: {value}"
    
    # FILTER OUT EMPTY EMPLOYEE RECORDS BEFORE PAGINATION
    filtered_rows = []
    for row in rows:
        row_dict = dict(zip(cols, row))
        empname = str(row_dict.get('empname', '')).strip()
        empemail = str(row_dict.get('empemail', '')).strip()
        
        # Only include rows with valid employee data
        if empname or empemail:
            filtered_rows.append(row)
    
    rows = filtered_rows
    
    if not rows:
        return "No valid employee records found matching your criteria."
    
    total_rows = len(rows)
    start_idx = (page_num - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    paginated_rows = rows[start_idx:end_idx]
    
    data_summary = format_data_for_verbalization(cols, paginated_rows, page_num, start_idx)
    
    estimated_tokens = len(paginated_rows) * 150 + 300
    max_tokens = min(estimated_tokens, 1500)
    
    # ENHANCED SYSTEM PROMPT WITH BETTER CONTEXT AWARENESS
    system_prompt = """You are a helpful assistant that converts database query results into natural, human-readable responses.

Your task:
1. Analyze the user's question and the data provided
2. Generate a clear, conversational response that answers the question
3. Present information in a natural way, avoiding technical jargon
4. Use proper formatting with bullet points for multiple items
5. Be concise but complete

**CRITICAL: UNDERSTAND THE QUERY TYPE**

1. **LEAVE BALANCE QUERIES** - When columns include 'cl', 'sl', 'co', or 'total_balance':
   - These show REMAINING/AVAILABLE leave days
   - Use phrases: "has X days remaining", "leave balance of X days", "X days available"
   - NEVER say "took X days" or "applied for X days"
   - Example: "John has 5.5 days of total leave balance remaining (CL: 2, SL: 2.5, CO: 1)"

2. **LEAVE HISTORY QUERIES** - When columns include 'from', 'to', 'leavetype', 'applied', 'reason':
   - These show leaves that were TAKEN/APPLIED
   - Use phrases: "took leave", "applied for leave", "was on leave"
   - Example: "John took 3 days of sick leave from Nov 1 to Nov 3"

3. **AGGREGATION QUERIES** - When columns include 'total_days', 'leave_count':
   - These show CUMULATIVE leave taken over a period
   - Use phrases: "took a total of X days", "applied for X days in total"
   - Example: "John took a total of 15 days across 3 leave applications this month"

**CRITICAL PAGINATION RULE:**
- You have been given a SMALL subset of results (typically 3-5 rows)
- You MUST include ALL rows provided in your response
- Each row must have its own bullet point
- Do NOT skip any rows
- SKIP rows where empname AND empemail are both empty/null

**DATA INTERPRETATION RULES:**
- If empname is empty but empemail exists, use empemail only
- If empemail is empty but empname exists, use empname only
- If BOTH empname and empemail are empty, SKIP that row entirely
- For leave balance values (cl, sl, co, total_balance): these are REMAINING days, not days taken
- For date ranges (from, to): these indicate when leave WAS taken
- For total_days with GROUP BY: this is cumulative days taken over a period

Guidelines:
- For single results: Give a direct answer
- For multiple results: List them clearly with relevant details (ONE bullet per row)
- For counts: State the number naturally
- For dates: Format them in a readable way (e.g., "from November 1st to November 5th")
- Always include employee names when available
- For leave balances: Mention the TYPE of balance (remaining/available, not taken)

**CORRECT EXAMPLES:**

Question: "list employees whose leave balance is less than 5"
Data: 
Row 1: empname: John, empemail: john@ex.com, total_balance: 3.5
Row 2: empname: Jane, empemail: jane@ex.com, total_balance: 4.0
Response: "Here are 2 employees with leave balance less than 5 days:

- John (john@ex.com) has 3.5 days of total leave balance remaining
- Jane (jane@ex.com) has 4.0 days of total leave balance remaining"

Question: "who has lowest casual leave balance"
Data: 
Row 1: empname: John, empemail: john@ex.com, cl: 1.5
Response: "John (john@ex.com) has the lowest casual leave balance with 1.5 days remaining."

Question: "employees who took more than 5 days this month"
Data:
Row 1: empname: John, empemail: john@ex.com, total_days: 8
Row 2: empname: Jane, empemail: jane@ex.com, total_days: 6
Response: "Here are employees who took more than 5 days of leave this month:

- John (john@ex.com) took a total of 8 days
- Jane (jane@ex.com) took a total of 6 days"

Question: "Who is on leave today?"
Data: [['John Doe', 'john@example.com', '2025-11-10', '2025-11-12', 'SICK LEAVE']]
Response: "John Doe (john@example.com) is currently on sick leave from November 10th to November 12th, 2025."

Question: "show employees with sick leave balance > 5"
Data: 
Row 1: empname: John, empemail: john@ex.com, sl: 18
Row 2: empname: Jane, empemail: jane@ex.com, sl: 12
Row 3: empname: , empemail: , sl: 8
Response: "Here are employees with sick leave balance greater than 5 days:

- John (john@ex.com) has 18 days of sick leave remaining
- Jane (jane@ex.com) has 12 days of sick leave remaining"

**INCORRECT EXAMPLES TO AVOID:**

❌ "John took 3.5 days" (when showing leave BALANCE)
❌ "Jane applied for 4 days" (when showing leave BALANCE)
❌ Including rows with empty empname and empemail
❌ "has 8 days remaining" (when showing TOTAL days taken in a period)

Now convert the following data into a natural response:"""

    # Detect query type for better user message
    q_lower = user_question.lower()
    cols_lower = [c.lower() for c in cols]
    
    query_type_hint = ""
    if any(col in cols_lower for col in ['cl', 'sl', 'co', 'total_balance']):
        if 'total_balance' in cols_lower or ('cl' in cols_lower and 'sl' in cols_lower and 'co' in cols_lower):
            query_type_hint = "\n**QUERY TYPE: Leave Balance (Remaining Days)**"
        else:
            query_type_hint = "\n**QUERY TYPE: Leave Balance (Specific Type)**"
    elif any(col in cols_lower for col in ['total_days', 'leave_count']) and 'from' not in cols_lower:
        query_type_hint = "\n**QUERY TYPE: Aggregated Leave History (Total Days Taken)**"
    elif any(col in cols_lower for col in ['from', 'to', 'leavetype', 'applied']):
        query_type_hint = "\n**QUERY TYPE: Leave History (Taken/Applied)**"

    user_message = f"""Question: {user_question}
{query_type_hint}

{data_summary}

IMPORTANT: 
1. Include ALL {len(paginated_rows)} rows in your response (one bullet per row)
2. Skip any rows where BOTH empname and empemail are empty
3. Use correct terminology based on query type (balance = remaining, history = taken)"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.1,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = out[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    response = re.sub(r"^(Response:|Answer:|A:)\s*", "", response, flags=re.IGNORECASE).strip()
    
    remaining = total_rows - end_idx
    if remaining > 0:
        response += f"\n\n📄 Showing results {start_idx + 1}-{end_idx} of {total_rows}. ({remaining} more remaining)"
    else:
        if total_rows > page_size:
            response += f"\n\n✓ Showing all {total_rows} results (page {page_num}/{(total_rows + page_size - 1) // page_size})."
    
    return response


def format_data_for_verbalization(cols: list, rows: list, page_num: int = 1, start_idx: int = 0) -> str:
    """Format database results for LLM verbalization."""
    if not rows:
        return "No data"
    
    formatted_lines = [
        f"⚠️ CRITICAL: You have {len(rows)} row(s) - MUST include ALL {len(rows)} in response",
        f"Columns: {', '.join(cols)}",
        ""
    ]
    
    for i, row in enumerate(rows, start=start_idx + 1):
        row_parts = []
        for col, val in zip(cols, row):
            if val is None:
                if any(keyword in col.lower() for keyword in ['days', 'count', 'total', 'balance', 'cl', 'sl', 'co', 'duration']):
                    row_parts.append(f"{col}: 0")
                else:
                    row_parts.append(f"{col}: None")
            elif isinstance(val, (int, float)):
                row_parts.append(f"{col}: {val}")
            elif isinstance(val, datetime.datetime):
                row_parts.append(f"{col}: {val.strftime('%Y-%m-%d %H:%M:%S')}")
            elif isinstance(val, datetime.date):
                row_parts.append(f"{col}: {val}")
            else:
                # Truncate very long text (like reasons)
                val_str = str(val)
                if len(val_str) > 100:
                    val_str = val_str[:97] + "..."
                row_parts.append(f"{col}: {val_str}")
        
        formatted_lines.append(f"Row {i}: {', '.join(row_parts)}")
    
    formatted_lines.append("")
    formatted_lines.append(f"⚠️ VERIFY: Count above = {len(rows)} rows. Your response MUST have {len(rows)} items.")
    
    return "\n".join(formatted_lines)

# --------------------- PAGINATION STATE ---------------------

class PaginationState:
    def __init__(self):
        self.last_query = None
        self.last_cols = None
        self.last_rows = None
        self.last_intent = None
        self.last_question = None
        self.current_page = 1
        self.page_size = 3

    def set_results(self, query, cols, rows, intent, question):
        self.last_query = query
        self.last_cols = cols
        self.last_rows = rows
        self.last_intent = intent
        self.last_question = question
        self.current_page = 1

    def next_page(self):
        if self.last_rows:
            max_page = (len(self.last_rows) + self.page_size - 1) // self.page_size
            if self.current_page < max_page:
                self.current_page += 1
                return True
        return False

    def has_data(self):
        return self.last_rows is not None and len(self.last_rows) > 0

    def reset(self):
        self.last_query = None
        self.last_cols = None
        self.last_rows = None
        self.last_intent = None
        self.last_question = None
        self.current_page = 1

# --------------------- FLASK ROUTES ---------------------

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'intent_detector_loaded': intent_detector is not None
    })

@app.route('/', methods=['GET'])
def home():
    try:
        return send_from_directory('.', 'index.html')
    except FileNotFoundError:
        return '''
        <h1>Error: index.html not found</h1>
        <p>Make sure index.html is in the same directory as app.py</p>
        <p><a href="/health">Check API Health</a></p>
        ''', 404

@app.route('/<path:path>')
def serve_static_files(path):
    try:
        return send_from_directory('.', path)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if model is None or intent_detector is None:
            return jsonify({'error': 'Models are still loading. Please wait...'}), 503

        data = request.json
        question = data.get('question', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        if session_id not in pagination_states:
            pagination_states[session_id] = PaginationState()
        
        pagination_state = pagination_states[session_id]
        
        if question.lower() in ['more', 'next', 'show more', 'continue']:
            if not pagination_state.has_data():
                return jsonify({
                    'answer': 'No previous query results to show more of. Please ask a new question.',
                    'has_more': False
                })
            
            if not pagination_state.next_page():
                return jsonify({
                    'answer': "You've reached the end of the results.",
                    'has_more': False
                })
            
            nl_response = verbalize_results(
                tokenizer, model,
                pagination_state.last_question,
                pagination_state.last_cols,
                pagination_state.last_rows,
                pagination_state.last_intent,
                page_size=pagination_state.page_size,
                page_num=pagination_state.current_page
            )
            
            total_rows = len(pagination_state.last_rows)
            max_page = (total_rows + pagination_state.page_size - 1) // pagination_state.page_size
            has_more = pagination_state.current_page < max_page
            
            return jsonify({
                'answer': nl_response,
                'has_more': has_more
            })
        
        pagination_state.reset()
        
        conn = get_db()
        if not conn.is_connected():
            return jsonify({'error': 'Database connection failed'}), 500
        conn = ensure_connection(conn)
        
        original_q = question
        question, date_context = preprocess_question(question)
        
        intent, confidence = intent_detector.detect(question)
        context = analyze_query_context(question)
        schema = get_schema_for_intent(intent)
        
        messages = build_llama_prompt(question, intent, context)
        raw_sql = generate_sql(tokenizer, model, messages)
        
        final_sql, is_valid, attempts = validate_and_repair_sql(
            conn, raw_sql, tokenizer, model, schema, question, intent, max_attempts=3
        )
        print(f"FINAL SQL: {final_sql}\n")
        
        cols, rows = run_query(conn, final_sql)
         # ADD COMPREHENSIVE DEBUG LOGGING
        print("="*60)
        print("DEBUG - VERBALIZATION INPUT")
        print("="*60)
        print(f"Question: {original_q}")
        print(f"Columns: {cols}")
        print(f"Total Rows: {len(rows)}")
        print(f"Rows data:")
        for i, row in enumerate(rows, 1):
            print(f"  Row {i}: {row}")
        print(f"Page size: {pagination_state.page_size}")
        print(f"Page num: {pagination_state.current_page}")
        print("="*60)
        pagination_state.set_results(final_sql, cols, rows, intent, original_q)
        
        nl_response = verbalize_results(
            tokenizer, model, original_q, cols, rows, intent,
            page_size=pagination_state.page_size,
            page_num=pagination_state.current_page
        )
        
        print("DEBUG - VERBALIZATION OUTPUT")
        print(f"Response: {nl_response}")
        print("="*60)
        
        total_rows = len(rows)
        max_page = (total_rows + pagination_state.page_size - 1) // pagination_state.page_size if total_rows > 0 else 1
        has_more = pagination_state.current_page < max_page
        
        safe_close_connection(conn)
        
        return jsonify({
            'answer': nl_response,
            'has_more': has_more
        })
        
    except Exception as e:
        print(f"Error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset_session', methods=['POST'])
def reset_session():
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        
        if session_id in pagination_states:
            pagination_states[session_id].reset()
        
        return jsonify({'message': 'Session reset successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --------------------- INITIALIZATION ---------------------

def initialize_models():
    global tokenizer, model, intent_detector
    print("="*60)
    print("INITIALIZING LEAVE CHATBOT FLASK APP")
    print("="*60)

    try:
        print("\n🔄 Loading Llama 3.1 8B model...")
        tokenizer, model = load_model()
        
        print("\n🔄 Loading intent detector...")
        intent_detector = IntentDetector()
        
        print("\n✅ All models loaded successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        print(traceback.format_exc())
        raise

# --------------------- MAIN ---------------------

if __name__ == '__main__':
    initialize_models()
    print("\n🚀 Starting Flask server...")
    print("📡 API will be available at: http://localhost:5000")
    print("📌 Endpoints:")
    print("   - GET / - Frontend interface")
    print("   - POST /chat - Main chatbot endpoint")
    print("   - POST /reset_session - Reset session state")
    print("   - GET /health - Health check")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

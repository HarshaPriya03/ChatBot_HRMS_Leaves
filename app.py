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

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# --------------------- GLOBAL VARIABLES ---------------------

tokenizer = None
model = None
intent_detector = None
pagination_states = {}  # Dictionary to store pagination state per session

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# --------------------- MODEL + DB (Same as original) ---------------------

def get_db():
    try:
        return mysql.connector.connect(
            host="68.178.155.255",
            user="Anika12",
            password="Anika12",
            database="ems",
            autocommit=True,
            pool_name="mypool",
            pool_size=5,
            pool_reset_session=True,
            connection_timeout=300  # 5 minutes timeout
        )
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        raise

def ensure_connection(conn):
    try:
        # Try to ping the connection to check if it's alive
        conn.ping(reconnect=True, attempts=3, delay=1)
        return conn
    except mysql.connector.Error:
        # If ping fails, get a new connection
        try:
            conn.close()
        except:
            pass  # Ignore errors when closing dead connection
        return get_db()

def safe_close_connection(conn):
    """Safely close a database connection without throwing errors"""
    try:
        if conn and conn.is_connected():
            conn.close()
    except Exception as e:
        print(f"Warning: Error closing connection: {e}")
        # Ignore errors when closing connection

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

# --------------------- DATE PREPROCESSING (Same as original) ---------------------

def preprocess_question(question: str) -> tuple:
    q_lower = question.lower()
    today = datetime.date.today()
    date_context = ""
    
    if "last month" in q_lower:
        last_month_start = (today.replace(day=1) - datetime.timedelta(days=1)).replace(day=1)
        last_month_end = today.replace(day=1) - datetime.timedelta(days=1)
        date_context = f"Date range: {last_month_start} to {last_month_end}"
        question = question.replace("last month", f"between '{last_month_start}' and '{last_month_end}'")
    
    elif "last 30 days" in q_lower or "past 30 days" in q_lower:
        date_30_days_ago = today - datetime.timedelta(days=30)
        date_context = f"Date range: {date_30_days_ago} to {today}"
        question = question.replace("last 30 days", f"after '{date_30_days_ago}'")
        question = question.replace("past 30 days", f"after '{date_30_days_ago}'")
    
    elif "this month" in q_lower:
        month_start = today.replace(day=1)
        date_context = f"Date range: {month_start} to {today}"
        question = question.replace("this month", f"from '{month_start}'")
    
    elif "this year" in q_lower:
        year_start = today.replace(month=1, day=1)
        date_context = f"Date range: {year_start} to {today}"
        question = question.replace("this year", f"after '{year_start}'")

    elif "day before yesterday" in q_lower:
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
    
    elif "last week" in q_lower:
        last_week_start = today - datetime.timedelta(days=today.weekday() + 7)
        last_week_end = last_week_start + datetime.timedelta(days=6)
        date_context = f"Date range: {last_week_start} to {last_week_end}"
        question = question.replace("last week", f"between '{last_week_start}' and '{last_week_end}'")
    
    return question, date_context

# --------------------- INTENT DETECTION (Enhanced) ---------------------

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
                "applied leaves", "leave history", "leaves taken", 
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
                "sick leave", "casual leave", "comp off",
                # ADDED: Enhanced aggregation examples
                "more than 3 days", "more than 5 days", "took more leaves",
                "more number of leaves", "took leaves more than", "exceeded leaves",
                "maximum leaves", "most leave days", "highest leave count",
                "employees with most leaves", "who took more", "top leave takers",
                "more leaves", "most number of leaves", "highest number of leaves",
                "employees took more leaves", "who took maximum leaves"
            ]
        }
        
        self.intent_embeds = {
            k: self.model.encode(v)
            for k, v in self.intent_examples.items()
        }
        print("Intent detector ready")
    
    def detect(self, question: str):
        q_emb = self.model.encode(question)
        
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
            elif any(word in q_lower for word in ['applied', 'history', 'latest', 'reason', 'when', 'days', 'who', 'phone', 'empph', 'contact', 'from', 'to', 'duration', 'took', 'type', 'sick', 'casual', 'comp']):
                return "leaves", 0.6
            return "unknown", best_score
        
        return best_intent, best_score

# --------------------- CONTEXT ANALYSIS (Enhanced) ---------------------

def analyze_query_context(question: str) -> dict:
    q_lower = question.lower()
    context = {
        'has_specific_email': False,
        'email': None,
        'is_general_query': False,
        'query_scope': 'unknown',
        'leave_type': None,
        'requires_aggregation': False,
        'aggregation_type': None,  # NEW: 'count', 'sum_days', 'duration'
        'min_days_threshold': None,
        'is_top_employees_query': False
    }

    specific_leave_keywords = ['casual leave', 'sick leave', 'comp off', 
                               'cl ', 'sl ', 'co ', 'casual', 'sick', 'comp']
    total_keywords = ['total', 'overall', 'combined', 'leave balance', 'all leaves']
    
    has_specific_type = any(keyword in q_lower for keyword in specific_leave_keywords)
    has_total_keyword = any(keyword in q_lower for keyword in total_keywords)
    
    if not has_specific_type or has_total_keyword:
        context['is_total_balance_query'] = True
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, question)
    if email_match:
        context['has_specific_email'] = True
        context['email'] = email_match.group()
        context['query_scope'] = 'specific_employee'
    
    if 'sick' in q_lower:
        context['leave_type'] = 'SICK LEAVE'
    elif 'casual' in q_lower:
        context['leave_type'] = 'CASUAL LEAVE'
    elif 'comp' in q_lower:
        context['leave_type'] = 'COMP OFF'
    
    if not context['has_specific_email']:
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
    
    # NEW: More flexible aggregation detection
    # Detect if query is about counting/summing leaves
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
    
    # Check if query requires aggregation
    for agg_type, keywords in aggregation_patterns.items():
        if any(keyword in q_lower for keyword in keywords):
            context['requires_aggregation'] = True
            context['aggregation_type'] = agg_type
            context['is_general_query'] = True
            context['query_scope'] = 'all_employees'
            break
    
    # Extract numeric thresholds for days
    days_match = re.search(r'(\d+)\s*days?', q_lower)
    if days_match:
        context['min_days_threshold'] = int(days_match.group(1))
    
    # Detect top N queries
    top_match = re.search(r'top\s+(\d+)\s+(employees|members)', q_lower)
    if top_match:
        context['is_top_employees_query'] = True
        context['top_limit'] = int(top_match.group(1))
    
    return context

# --------------------- QUICK SYNTAX FIXES (Enhanced) ---------------------

def quick_syntax_fix(sql: str) -> str:
    # Remove NULLS FIRST/LAST (not supported in MySQL)
    sql = re.sub(r'\s+NULLS\s+(FIRST|LAST)', '', sql, flags=re.IGNORECASE)
    
    # Replace ILIKE with LIKE
    sql = re.sub(r'\bILIKE\b', 'LIKE', sql, flags=re.IGNORECASE)
    
    # Fix common column name mistakes
    sql = re.sub(r'\bltype\b', 'leavetype', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bemail\b(?![\w@])', 'empemail', sql, flags=re.IGNORECASE)
    
    # Fix DATEDIFF syntax (PostgreSQL → MySQL)
    sql = re.sub(
        r"DATEDIFF\s*\(\s*['\"]day['\"]\s*,\s*([^,]+)\s*,\s*([^)]+)\)",
        r"DATEDIFF(\2, \1)",
        sql,
        flags=re.IGNORECASE
    )
    
    # Fix INTERVAL syntax
    sql = re.sub(
        r"INTERVAL\s+'(\d+)\s+(day|week|month|year)s?'",
        r"INTERVAL \1 \2",
        sql,
        flags=re.IGNORECASE
    )
    
    # Fix CURRENT_DATE - INTERVAL syntax
    sql = re.sub(
        r'\(?\s*CURRENT_DATE\s*-\s*INTERVAL\s+(\d+)\s+(DAY|WEEK|MONTH|YEAR)\s*\)?',
        r'DATE_SUB(CURDATE(), INTERVAL \1 \2)',
        sql,
        flags=re.IGNORECASE
    )
    
    # Replace CURRENT_DATE with CURDATE()
    sql = re.sub(r'\bCURRENT_DATE\(\)\b', 'CURDATE()', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bCURRENT_DATE\b', 'CURDATE()', sql, flags=re.IGNORECASE)
    
    # Add backticks to reserved words 'from' and 'to' when used as column names
    sql = re.sub(r'\bfrom\b(?![\s`])', '`from`', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bto\b(?![\s`])', '`to`', sql, flags=re.IGNORECASE)
    
    # Remove redundant BETWEEN conditions when using CURDATE()
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
    
    # ✅ FIX ORDER BY for cl/sl/co columns (only if not already CAST)
    if not re.search(r'ORDER\s+BY\s+CAST\s*\(\s*(cl|sl|co)', sql, flags=re.IGNORECASE):
        sql = re.sub(
            r'\bORDER\s+BY\s+(cl|sl|co)\b(\s+(ASC|DESC))?',
            lambda m: f'ORDER BY CAST({m.group(1)} AS DECIMAL(10,2)){m.group(2) if m.group(2) else ""}',
            sql,
            flags=re.IGNORECASE
        )
    
    # Fix SUM(DATEDIFF()) patterns if needed
    sql = re.sub(
        r'SUM\s*\(\s*DATEDIFF\s*\(\s*`to`\s*,\s*`from`\s*\)\s*\+\s*1\s*\)',
        'SUM(DATEDIFF(`to`, `from`) + 1)',
        sql,
        flags=re.IGNORECASE
    )
    
    # ✅ REMOVED: The problematic CAST conversion for WHERE clauses
    # This was causing the syntax error by converting too many patterns
    
    # Fix leavetype comparisons to use LIKE instead of =
    sql = re.sub(
        r"leavetype\s*=\s*'(sick|casual|comp)'",
        r"LOWER(leavetype) LIKE '%\1%'",
        sql,
        flags=re.IGNORECASE
    )
    
    # Add DISTINCT for phone queries
    if re.search(r'SELECT\s+empph\s+FROM', sql, flags=re.IGNORECASE) and \
       not re.search(r'SELECT\s+DISTINCT', sql, flags=re.IGNORECASE):
        sql = re.sub(r'SELECT\s+empph', 'SELECT DISTINCT empph', sql, flags=re.IGNORECASE)
    
    return sql

# --------------------- SCHEMA TEMPLATES (Enhanced) ---------------------

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
- For latest leavebalance/total balance: CAST(cl AS DECIMAL) + CAST(sl AS DECIMAL) + CAST(co AS DECIMAL)
- **CRITICAL: When user asks for "leave balance less than X" or "total balance less than X", 
  calculate TOTAL: (CAST(cl AS DECIMAL) + CAST(sl AS DECIMAL) + CAST(co AS DECIMAL)) < X**
- **NEVER use OR conditions for total leave balance queries**
""",
    
    "leaves": """
Table: leaves
Columns:
  - ID: PRIMARY KEY
  - empname: VARCHAR (employee name)
  - empemail: VARCHAR (employee email) - CRITICAL: Use 'empemail' NOT 'email'
  - leavetype: VARCHAR ('SICK LEAVE', 'CASUAL LEAVE', 'COMP OFF') - Use 'leavetype' NOT 'ltype'
  - applied: DATETIME (when leave was submitted)
  - from: DATE (leave start date) - MUST use backticks
  - to: DATE (leave end date) - MUST use backticks
  - desg: VARCHAR (designation)
  - reason: TEXT (leave reason)
  - empph: VARCHAR (employee phone)
  - work_location: VARCHAR

CRITICAL DATE QUERY RULES:
- For "on leave today" → WHERE CURDATE() BETWEEN from AND to
- For "on leave on specific date" → WHERE '2025-11-06' BETWEEN from AND to
- NEVER mix CURDATE() with specific date checks in same query
- For "latest/recent leave" → ORDER BY from DESC LIMIT 1
- For "latest [SICK/CASUAL/COMP] leave" → WHERE LOWER(leavetype) LIKE '%sick%' ORDER BY from DESC LIMIT 1
- For "when applied" → Use 'applied' column, ORDER BY applied DESC
- For duration: DATEDIFF(to, from) + 1
- For phone: SELECT DISTINCT empph LIMIT 1
- Column is 'leavetype' NOT 'ltype'
- Use LOWER(leavetype) LIKE '%keyword%' for case-insensitive matching

**CRITICAL: TOTAL LEAVE DAYS CALCULATION**
When user asks about employees who took "more than X days" OR "more leaves" OR "most leaves":
1. Calculate TOTAL days per employee: SUM(DATEDIFF(`to`, `from`) + 1)
2. Group by empname, empemail
3. Use HAVING clause to filter: HAVING SUM(DATEDIFF(`to`, `from`) + 1) > X
4. Order by total days DESC for rankings
5. ALWAYS include the calculated total in SELECT as 'total_days'

EXAMPLES:
- "employees who took more than 5 days this month" → 
  SELECT empname, empemail, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days 
  FROM leaves WHERE `from` >= '2025-11-01' 
  GROUP BY empname, empemail 
  HAVING total_days > 5 
  ORDER BY total_days DESC

- "top 5 employees with most leaves this month" →
  SELECT empname, empemail, COUNT(*) AS leave_count, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days 
  FROM leaves WHERE `from` >= '2025-11-01' 
  GROUP BY empname, empemail 
  ORDER BY total_days DESC LIMIT 5
"""
}

def get_schema_for_intent(intent: str, original_question: str = "") -> str:
    q_lower = original_question.lower() if original_question else "" 
    if intent == "leavebalance":
        if "can" in q_lower and "apply" in q_lower:
            intent_hint = "TASK: Check if employee can apply for specified leave type by verifying balance > 0"
        else:
            intent_hint = "TASK: Query 'leavebalance' table. Return cl, sl, co values."
    elif intent == "leaves":
        return SCHEMA_TEMPLATES["leaves"]
    else:
        return SCHEMA_TEMPLATES["leavebalance"] + "\n" + SCHEMA_TEMPLATES["leaves"]

# --------------------- LLAMA 3.1 PROMPT BUILDING (Enhanced) ---------------------

def build_llama_prompt(user_question: str, intent: str, context: dict) -> str:
    schema = get_schema_for_intent(intent)
    
    context_hint = ""
    if context['has_specific_email']:
        context_hint = f"FILTER: WHERE empemail = '{context['email']}'"
    elif context['is_general_query']:
        context_hint = "FILTER: Query ALL employees. DO NOT filter by specific email. Include empname and empemail in SELECT."
    
    q_lower = user_question.lower()
    intent_hint = ""
    
    if intent == "leavebalance":
        if context.get('is_total_balance_query', False):
            if any(word in q_lower for word in ['less than', '<', 'below', 'under']):
                intent_hint = """TASK: Query 'leavebalance' table for TOTAL leave balance.
Calculate: (CAST(cl AS DECIMAL(10,2)) + CAST(sl AS DECIMAL(10,2)) + CAST(co AS DECIMAL(10,2))) AS total_balance
Use WHERE clause with the same calculation.
Include empname, empemail, cl, sl, co, total_balance in SELECT."""
            elif any(word in q_lower for word in ['greater than', '>', 'above', 'more than']):
                intent_hint = "TASK: Query 'leavebalance' table for TOTAL balance > threshold. Use same calculation."
        elif context['leave_type']:
            leave_col = 'cl' if 'CASUAL' in context['leave_type'] else 'sl' if 'SICK' in context['leave_type'] else 'co'
            intent_hint = f"TASK: Query 'leavebalance' table. Filter by {leave_col} only. Use CAST({leave_col} AS DECIMAL(10,2)) in WHERE."
        elif "can" in q_lower and "apply" in q_lower:
            intent_hint = "TASK: Check if employee can apply for specified leave type by verifying balance > 0"
        else:
            intent_hint = "TASK: Query 'leavebalance' table. Return cl, sl, co values."
    
    elif intent == "leaves":
        # NEW: Unified aggregation handling
        if context.get('requires_aggregation', False):
            # Extract numeric thresholds
            days_threshold = context.get('min_days_threshold')
            
            top_match = re.search(r'top\s*(\d+)', q_lower)
            limit_count = top_match.group(1) if top_match else '5'
            
            # Determine aggregation type
            agg_type = context.get('aggregation_type', 'sum_days')
            
            # Build appropriate hint based on query intent
            if agg_type == 'count':
                # User wants COUNT of employees
                if days_threshold:
                    intent_hint = f"""TASK: COUNT how many employees took more than {days_threshold} total days.
Structure: SELECT COUNT(*) FROM (subquery with GROUP BY and HAVING)
SELECT COUNT(*) FROM (
    SELECT empemail FROM leaves 
    WHERE [date filter]
    GROUP BY empemail 
    HAVING SUM(DATEDIFF(`to`, `from`) + 1) > {days_threshold}
) AS subquery;"""
                else:
                    intent_hint = """TASK: COUNT employees who took leaves.
Use COUNT(DISTINCT empemail) or subquery with GROUP BY."""
            
            elif agg_type == 'top_ranking' or 'top' in q_lower:
                # User wants top N employees
                intent_hint = f"""TASK: Find top {limit_count} employees with MOST total leave days.
CRITICAL RULES:
1. Calculate TOTAL days per employee: SUM(DATEDIFF(`to`, `from`) + 1)
2. Include both leave_count and total_days
3. GROUP BY empname, empemail
4. ORDER BY total_days DESC (NOT leave_count)
5. LIMIT {limit_count}

Structure:
SELECT empname, empemail, COUNT(*) AS leave_count, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days
FROM leaves
WHERE [date filter if specified]
GROUP BY empname, empemail
ORDER BY total_days DESC
LIMIT {limit_count};"""
            
            else:
                # Default: sum_days type (most common case)
                comparison_operators = {
                    'more than': '>', 'greater than': '>', 'over': '>', 'above': '>',
                    'less than': '<', 'below': '<', 'under': '<', 'fewer than': '<'
                }
                
                operator = '>'  # default
                for phrase, op in comparison_operators.items():
                    if phrase in q_lower:
                        operator = op
                        break
                
                if days_threshold:
                    intent_hint = f"""TASK: Find employees who took {operator} {days_threshold} TOTAL days of leave.
CRITICAL RULES:
1. Calculate TOTAL days across ALL leave applications per employee
2. Use SUM(DATEDIFF(`to`, `from`) + 1) AS total_days
3. GROUP BY empname, empemail
4. Use HAVING clause: HAVING total_days {operator} {days_threshold}
5. ORDER BY total_days DESC
6. Include total_days in SELECT

Structure:
SELECT empname, empemail, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days
FROM leaves
WHERE [date filter if specified]
GROUP BY empname, empemail
HAVING total_days {operator} {days_threshold}
ORDER BY total_days DESC;"""
                else:
                    # Generic "most leaves" or "more leaves" without specific threshold
                    intent_hint = """TASK: Find employees with MOST total leave days (no specific threshold).
CRITICAL RULES:
1. Calculate TOTAL days: SUM(DATEDIFF(`to`, `from`) + 1) AS total_days
2. Include COUNT(*) AS leave_count to show number of applications
3. GROUP BY empname, empemail
4. ORDER BY total_days DESC (this shows who took most)
5. DO NOT use LIMIT unless specifically asked for "top N"

Structure:
SELECT empname, empemail, COUNT(*) AS leave_count, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days
FROM leaves
WHERE [date filter if specified]
GROUP BY empname, empemail
ORDER BY total_days DESC;"""
        
        # Existing non-aggregation handlers
        elif any(word in q_lower for word in ['latest', 'recent', 'last']):
            leave_type_filter = ""
            if context['leave_type']:
                leave_type_filter = f" AND LOWER(leavetype) LIKE '%{context['leave_type'].split()[0].lower()}%'"
            
            if context['has_specific_email']:
                intent_hint = f"TASK: Use 'leaves' table. Find most recent leave{leave_type_filter} for {context['email']}. ORDER BY `from` DESC LIMIT 1. Show `from`, `to`, leavetype, reason."
            else:
                intent_hint = f"TASK: Use 'leaves' table. Find most recent leave{leave_type_filter} across ALL employees (no email filter). ORDER BY `from` DESC LIMIT 1. Show empname, empemail, `from`, `to`, leavetype."
        
        elif any(word in q_lower for word in ['duration', 'how many days', 'how long']):
            intent_hint = "TASK: Calculate duration using DATEDIFF(`to`, `from`) + 1"
        
        elif "phone" in q_lower or "contact" in q_lower:
            intent_hint = "TASK: SELECT DISTINCT empph (phone number) from leaves table LIMIT 1"
        
        elif any(word in q_lower for word in ['on leave', 'who is on leave', 'members on leave']):
            date_pattern = r"on\s+'([\d-]+)'"
            date_match = re.search(date_pattern, user_question)
            if date_match:
                specific_date = date_match.group(1)
                intent_hint = f"TASK: Find employees on leave on {specific_date}. WHERE '{specific_date}' BETWEEN `from` AND `to`. DO NOT use CURDATE()."
            else:
                intent_hint = "TASK: Find employees currently on leave. WHERE CURDATE() BETWEEN `from` AND `to`"
        
        elif any(word in q_lower for word in ['top', 'most', 'maximum']):
            intent_hint = "TASK: GROUP BY empemail, COUNT leaves, ORDER BY count DESC, use LIMIT"
    
    warning = ""
    if context['is_general_query'] or not context['has_specific_email']:
        warning = "\n⚠️ CRITICAL WARNING: Do NOT use 'john@example.com' or any example emails from below. The query should work for ALL employees without email filtering.\n"
    
    system_prompt = f"""You are a MySQL query generator for an Employee Management System (EMS) database.

DATABASE SCHEMA:
{schema}

MYSQL SYNTAX REQUIREMENTS:
- Use ONLY 'leavebalance' and 'leaves' tables
- Column names: 'empemail' (NOT 'email'), 'leavetype' (NOT 'ltype')
- Reserved words need backticks: `from`, `to`
- Use CURDATE() not CURRENT_DATE
- DATEDIFF(end, start) - 2 parameters only
- For case-insensitive: LOWER(column) LIKE '%value%'
- Date format: 'YYYY-MM-DD'
- **CRITICAL: cl/sl/co are VARCHAR, use CAST(column AS DECIMAL(10,2)) in ORDER BY for min/max**

{context_hint}
{intent_hint}
{warning}

CRITICAL EXAMPLES:
Q: "what is leave balance/ latest leave balance/ latest leavebalance of john@example.com"
A: SELECT cl, sl, co FROM leavebalance WHERE empemail = 'john@example.com';

Q: "who has the lowest casual leave balance"
A: SELECT empname, empemail, cl FROM leavebalance ORDER BY CAST(cl AS DECIMAL(10,2)) ASC LIMIT 1;

Q: "who has the highest casual leave balance"
A: SELECT empname, empemail, cl FROM leavebalance ORDER BY CAST(cl AS DECIMAL(10,2)) DESC LIMIT 1;

Q: "who has the lowest sick leave balance"
A: SELECT empname, empemail, sl FROM leavebalance ORDER BY CAST(sl AS DECIMAL(10,2)) ASC LIMIT 1;

Q: "what is the leave duration of putsalaharshapriya@gmail.com in her last leave?" 
A: SELECT DATEDIFF(`to`, `from`) + 1 FROM leaves WHERE empemail = 'putsalaharshapriya@gmail.com' ORDER BY `from` DESC LIMIT 1;

Q: "top 5 employees with highest comp off balance"
A: SELECT empname, empemail, co FROM leavebalance ORDER BY CAST(co AS DECIMAL(10,2)) DESC LIMIT 5;

Q: "latest leave of john@example.com"
A: SELECT empname, `from`, `to`, leavetype, reason FROM leaves WHERE empemail = 'john@example.com' ORDER BY `from` DESC LIMIT 1;

Q: "latest sick leave of john@example.com"
A: SELECT empname, `from`, `to`, leavetype, reason FROM leaves WHERE empemail = 'john@example.com' AND LOWER(leavetype) LIKE '%sick%' ORDER BY `from` DESC LIMIT 1;

Q: "what is the latest leave applied"
A: SELECT empname, empemail, `from`, `to`, leavetype, applied FROM leaves ORDER BY applied DESC LIMIT 1;

Q: "what is the latest leave record"
A: SELECT empname, empemail, `from`, `to`, leavetype, reason FROM leaves ORDER BY `from` DESC LIMIT 1;

Q: "who is on leave today"
A: SELECT empname, empemail, `from`, `to`, leavetype FROM leaves WHERE CURDATE() BETWEEN `from` AND `to`;

Q: "who is on leave on '2025-11-06'"
A: SELECT empname, empemail, `from`, `to`, leavetype FROM leaves WHERE '2025-11-06' BETWEEN `from` AND `to`;

Q: "list of employees who are on leave on '2025-11-05'"
A: SELECT empname, empemail, `from`, `to`, leavetype FROM leaves WHERE '2025-11-05' BETWEEN `from` AND `to`;

Q: "how many members are on leave on '2025-11-07'"
A: SELECT COUNT(*) FROM leaves WHERE '2025-11-07' BETWEEN `from` AND `to`;

Q: "can john@example.com apply for Casual Leave"
A: SELECT CASE WHEN CAST(cl AS DECIMAL(10,2)) > 0 THEN 'Yes' ELSE 'No' END AS can_apply, cl FROM leavebalance WHERE empemail = 'john@example.com';

Q: "top 5 employees who took most leaves"
A: SELECT empname, empemail, COUNT(*) AS leave_count FROM leaves GROUP BY empname, empemail ORDER BY leave_count DESC LIMIT 5;

Q: "phone number of john@example.com"
A: SELECT DISTINCT empph FROM leaves WHERE empemail = 'john@example.com' LIMIT 1;

**NEW AGGREGATION EXAMPLES (TOTAL DAYS CALCULATION):**

Q: "give me the list of employees who took more than 5 days in this month"
A: SELECT empname, empemail, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days FROM leaves WHERE `from` >= DATE_FORMAT(CURDATE(), '%Y-%m-01') GROUP BY empname, empemail HAVING total_days > 5 ORDER BY total_days DESC;

Q: "employees who took more leaves in this month"
A: SELECT empname, empemail, COUNT(*) AS leave_count, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days FROM leaves WHERE `from` >= DATE_FORMAT(CURDATE(), '%Y-%m-01') GROUP BY empname, empemail ORDER BY total_days DESC;

Q: "give me the employee names who took leaves more leaves in this month"
A: SELECT empname, empemail, COUNT(*) AS leave_count, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days FROM leaves WHERE `from` >= DATE_FORMAT(CURDATE(), '%Y-%m-01') GROUP BY empname, empemail ORDER BY total_days DESC;

Q: "employee names who took more leaves this month"
A: SELECT empname, empemail, COUNT(*) AS leave_count, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days FROM leaves WHERE `from` >= DATE_FORMAT(CURDATE(), '%Y-%m-01') GROUP BY empname, empemail ORDER BY total_days DESC;

Q: "list employees took most leaves this month"
A: SELECT empname, empemail, COUNT(*) AS leave_count, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days FROM leaves WHERE `from` >= DATE_FORMAT(CURDATE(), '%Y-%m-01') GROUP BY empname, empemail ORDER BY total_days DESC;

Q: "show me employees with maximum leaves this month"
A: SELECT empname, empemail, COUNT(*) AS leave_count, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days FROM leaves WHERE `from` >= DATE_FORMAT(CURDATE(), '%Y-%m-01') GROUP BY empname, empemail ORDER BY total_days DESC;

Q: "who took highest number of leaves this month"
A: SELECT empname, empemail, COUNT(*) AS leave_count, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days FROM leaves WHERE `from` >= DATE_FORMAT(CURDATE(), '%Y-%m-01') GROUP BY empname, empemail ORDER BY total_days DESC LIMIT 1;

Q: "how many members took leave more than 3 days in this month"
A: SELECT COUNT(*) FROM (SELECT empemail FROM leaves WHERE `from` >= DATE_FORMAT(CURDATE(), '%Y-%m-01') GROUP BY empemail HAVING SUM(DATEDIFF(`to`, `from`) + 1) > 3) AS subquery;

Q: "top 5 employees who took most leaves this month"
A: SELECT empname, empemail, COUNT(*) AS leave_count, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days FROM leaves WHERE `from` >= DATE_FORMAT(CURDATE(), '%Y-%m-01') GROUP BY empname, empemail ORDER BY total_days DESC LIMIT 5;

Q: "give me employee records who took leave more than 3 days in this month"
A: SELECT empname, empemail, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days FROM leaves WHERE `from` >= DATE_FORMAT(CURDATE(), '%Y-%m-01') GROUP BY empname, empemail HAVING total_days > 3 ORDER BY total_days DESC;

Q: "list of employees who took more number of leaves in this month"
A: SELECT empname, empemail, COUNT(*) AS leave_count, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days FROM leaves WHERE `from` >= DATE_FORMAT(CURDATE(), '%Y-%m-01') GROUP BY empname, empemail ORDER BY total_days DESC;

Q: "give me employee names who took leaves more than 3 days in this month"
A: SELECT empname, empemail, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days FROM leaves WHERE `from` >= DATE_FORMAT(CURDATE(), '%Y-%m-01') GROUP BY empname, empemail HAVING total_days > 3 ORDER BY total_days DESC;

Q: "employees who took more than 5 days last month"
A: SELECT empname, empemail, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days FROM leaves WHERE `from` >= DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 MONTH), '%Y-%m-01') AND `from` < DATE_FORMAT(CURDATE(), '%Y-%m-01') GROUP BY empname, empemail HAVING total_days > 5 ORDER BY total_days DESC;

Q: "top 3 employees with most leaves this year"
A: SELECT empname, empemail, COUNT(*) AS leave_count, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days FROM leaves WHERE YEAR(`from`) = YEAR(CURDATE()) GROUP BY empname, empemail ORDER BY total_days DESC LIMIT 3;

Q: "employees who took more leaves last 30 days"
A: SELECT empname, empemail, COUNT(*) AS leave_count, SUM(DATEDIFF(`to`, `from`) + 1) AS total_days FROM leaves WHERE `from` >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) GROUP BY empname, empemail ORDER BY total_days DESC;

CRITICAL DATE RANGE RULES:
1. When asked for "leave records between DATE1 and DATE2" OR "last month records" OR "records from DATE1 to DATE2":
   → Filter by leave start date: WHERE `from` BETWEEN 'DATE1' AND 'DATE2'
   
2. When asked "who is on leave on DATE" OR "members on leave on DATE" OR "who applied leave on DATE":
   → Find leaves that include that date: WHERE 'DATE' BETWEEN `from` AND `to`
   
3. When asked "who is on leave today":
   → Use: WHERE CURDATE() BETWEEN `from` AND `to`

Now generate MySQL query for:"""

    user_message = f"Question: {user_question}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    return messages

# --------------------- SQL GENERATION (Same as original) ---------------------

def generate_sql(tokenizer, model, messages: list, max_new_tokens: int = 400) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
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
    
    return fixed_sql

# --------------------- LLM REPAIR (Same as original) ---------------------

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
        repair_hint = "ERROR: 'from' and 'to' are reserved keywords. Use backticks: from, to"
        sql = re.sub(r'\bfrom\b(?![`\s])', 'from', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bto\b(?![`\s])', 'to', sql, flags=re.IGNORECASE)
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

# --------------------- VALIDATION & EXECUTION (Same as original) ---------------------

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

# --------------------- VERBALIZATION (Same as original) ---------------------

def verbalize_results(tokenizer, model, user_question: str, cols: list, rows: list, intent: str, page_size: int = 3, page_num: int = 1) -> str:
    if not rows:
        return "No results found for your query."
    
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
    
    total_rows = len(rows)
    start_idx = (page_num - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    paginated_rows = rows[start_idx:end_idx]
    
    data_summary = format_data_for_verbalization(cols, paginated_rows, page_num, start_idx)
    
    estimated_tokens = len(paginated_rows) * 150 + 300
    max_tokens = min(estimated_tokens, 1500)
    
    system_prompt = """You are a helpful assistant that converts database query results into natural, human-readable responses.

Your task:
1. Analyze the user's question and the data provided
2. Generate a clear, conversational response that answers the question
3. Present information in a natural way, avoiding technical jargon
4. Use proper formatting with bullet points for multiple items
5. Be concise but complete

**CRITICAL PAGINATION RULE:**
- You have been given a SMALL subset of results (typically 3-5 rows)
- You MUST include ALL rows provided in your response
- Each row must have its own bullet point
- Do NOT skip any rows

Guidelines:
- For single results: Give a direct answer
- For multiple results: List them clearly with relevant details (ONE bullet per row)
- For counts: State the number naturally
- For dates: Format them in a readable way (e.g., "from November 1st to November 5th")
- Always include employee names when available
- For leave balances: Mention Casual Leave (CL), Sick Leave (SL), and Comp Off (CO) clearly

Examples:
Question: "Who is on leave today?"
Data: [['John Doe', 'john@example.com', '2025-11-10', '2025-11-12', 'SICK LEAVE']]
Response: "John Doe (john@example.com) is currently on sick leave from November 10th to November 12th, 2025."

Question: "show employees with sick leave > 5"
Data: 
Row 1: empname: John, empemail: john@ex.com, sl: 18
Row 2: empname: Jane, empemail: jane@ex.com, sl: 12
Row 3: empname: Bob, empemail: bob@ex.com, sl: 8
Response: "Here are employees with sick leave balance greater than 5:

- John (john@ex.com) with 18 days of sick leave
- Jane (jane@ex.com) with 12 days of sick leave
- Bob (bob@ex.com) with 8 days of sick leave"

Now convert the following data into a natural response:"""

    user_message = f"""Question: {user_question}

{data_summary}

IMPORTANT: Include ALL {len(paginated_rows)} rows in your response. Each row should have its own bullet point."""

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
    if not rows:
        return "No data"
    
    formatted_lines = [f"Columns: {', '.join(cols)}\n"]
    
    for i, row in enumerate(rows, start=start_idx + 1):
        row_data = []
        for col, val in zip(cols, row):
            if val is not None:
                row_data.append(f"{col}: {val}")
            else:
                row_data.append(f"{col}: NULL")
        formatted_lines.append(f"Row {i}: {', '.join(row_data)}")
    
    return "\n".join(formatted_lines)

# --------------------- PAGINATION STATE (Same as original) ---------------------

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

# --------------------- FLASK ROUTES (CLEANED RESPONSE) ---------------------

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'intent_detector_loaded': intent_detector is not None
    })

@app.route('/', methods=['GET'])
def home():
    """Serve the frontend HTML"""
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
    """Serve static files (CSS, JS, etc.)"""
    try:
        return send_from_directory('.', path)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint - CLEANED RESPONSE"""
    try:
        # Check if models are loaded
        if model is None or intent_detector is None:
            return jsonify({'error': 'Models are still loading. Please wait...'}), 503
        
        data = request.json
        question = data.get('question', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Get or create pagination state for this session
        if session_id not in pagination_states:
            pagination_states[session_id] = PaginationState()
        
        pagination_state = pagination_states[session_id]
        
        # Check if user wants more results
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
            
            # Generate response for next page
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
        
        # Reset pagination for new query
        pagination_state.reset()
        
        # Get database connection
        conn = get_db()
        if not conn.is_connected():
            return jsonify({'error': 'Database connection failed'}), 500
        conn = ensure_connection(conn)
        
        # Preprocess question for dates
        original_q = question
        question, date_context = preprocess_question(question)
        
        # Detect intent
        intent, confidence = intent_detector.detect(question)
        
        # Analyze context
        context = analyze_query_context(question)
        
        # Get schema
        schema = get_schema_for_intent(intent)
        
        # Build prompt and generate SQL
        messages = build_llama_prompt(question, intent, context)
        raw_sql = generate_sql(tokenizer, model, messages)
        
        # Validate and repair SQL
        final_sql, is_valid, attempts = validate_and_repair_sql(
            conn, raw_sql, tokenizer, model, schema, question, intent, max_attempts=3
        )
        print(f"final {final_sql}\n")
        # Execute query
        cols, rows = run_query(conn, final_sql)
        
        # Store results for pagination
        pagination_state.set_results(final_sql, cols, rows, intent, original_q)
        
        # Generate natural language response
        nl_response = verbalize_results(
            tokenizer, model, original_q, cols, rows, intent,
            page_size=pagination_state.page_size,
            page_num=pagination_state.current_page
        )
        
        # Calculate pagination info
        total_rows = len(rows)
        max_page = (total_rows + pagination_state.page_size - 1) // pagination_state.page_size if total_rows > 0 else 1
        has_more = pagination_state.current_page < max_page
        
        safe_close_connection(conn)
        
        # RETURN ONLY CLEAN RESPONSE - NO TECHNICAL DETAILS
        return jsonify({
            'answer': nl_response,
            'has_more': has_more
        })
        
    except Exception as e:
        print(f"Error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset pagination state for a session"""
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
    """Initialize models on startup"""
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
    # Initialize models before starting the server
    initialize_models()
    
    # Start Flask server
    print("\n🚀 Starting Flask server...")
    print("📡 API will be available at: http://localhost:5000")
    print("📌 Endpoints:")
    print("   - GET / - Frontend interface")
    print("   - POST /chat - Main chatbot endpoint")
    print("   - POST /reset_session - Reset session state")
    print("   - GET /health - Health check")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

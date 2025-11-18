import mysql.connector
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import datetime
from sentence_transformers import SentenceTransformer, util

# --------------------- MODEL + DB ---------------------

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def get_db():
    return mysql.connector.connect(
        host="68.178.155.255",
        user="Anika12",
        password="Anika12",
        database="ems",
        autocommit=True,
        pool_name="mypool",
        pool_size=3,
        pool_reset_session=True
    )

def ensure_connection(conn):
    try:
        conn.ping(reconnect=True, attempts=3, delay=1)
        return conn
    except mysql.connector.Error:
        return get_db()

def load_model():
    print("Loading Llama 3.1 8B Instruct in 4-bit...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization config for bitsandbytes
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

# --------------------- DATE PREPROCESSING ---------------------

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
                "sick leave", "casual leave", "comp off"
            ]
        }
        
        self.intent_embeds = {
            k: self.model.encode(v, convert_to_tensor=True) 
            for k, v in self.intent_examples.items()
        }
        print("Intent detector ready")
    
    def detect(self, question: str):
        q_emb = self.model.encode(question, convert_to_tensor=True)
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

# --------------------- CONTEXT ANALYSIS ---------------------

def analyze_query_context(question: str) -> dict:
    q_lower = question.lower()
    context = {
        'has_specific_email': False,
        'email': None,
        'is_general_query': False,
        'query_scope': 'unknown',
        'leave_type': None
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
        person_plurals = ['employees', 'members', 'people', 'persons', 'staff']
        quantifiers = ['all', 'any', 'every', 'everyone', 'anyone', 'maximum', 'most']
        
        has_question_word = any(word in q_lower for word in question_indicators)
        has_plural = any(word in q_lower for word in person_plurals)
        has_quantifier = any(word in q_lower for word in quantifiers)
        
        if has_question_word or has_plural or has_quantifier:
            context['is_general_query'] = True
            context['query_scope'] = 'all_employees'
        else:
            context['query_scope'] = 'unclear'
    
    return context

# --------------------- QUICK SYNTAX FIXES ---------------------

def quick_syntax_fix(sql: str) -> str:
    sql = re.sub(r'\s+NULLS\s+(FIRST|LAST)', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bILIKE\b', 'LIKE', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bltype\b', 'leavetype', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bemail\b(?![\w@])', 'empemail', sql, flags=re.IGNORECASE)
    
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
    
    sql = re.sub(r'\bfrom\b(?![\s`])', 'from', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bto\b(?![\s`])', 'to', sql, flags=re.IGNORECASE)
    
    if 'CURDATE() BETWEEN' in sql.upper():
        sql = re.sub(
            r"\s+AND\s+from\s*<=\s*['\"][\d-]+['\"]",
            "",
            sql,
            flags=re.IGNORECASE
        )
        sql = re.sub(
            r"\s+AND\s+to\s*>=\s*['\"][\d-]+['\"]",
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
        r'\b(cl|sl|co)\s*([><=]+)\s*([0-9.-]+)',
        r'CAST(\1 AS DECIMAL(10,2)) \2 \3',
        sql,
        flags=re.IGNORECASE
    )

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

# --------------------- SCHEMA TEMPLATES ---------------------

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

# --------------------- LLAMA 3.1 PROMPT BUILDING ---------------------

def build_llama_prompt(user_question: str, intent: str, context: dict) -> str:
    schema = get_schema_for_intent(intent)
    
    # Build context hints
    context_hint = ""
    if context['has_specific_email']:
        context_hint = f"FILTER: WHERE empemail = '{context['email']}'"
    elif context['is_general_query']:
        context_hint = "FILTER: Query ALL employees. DO NOT filter by specific email. Include empname and empemail in SELECT."
    
    # Add intent-specific hints
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
            # Specific leave type query
            leave_col = 'cl' if 'CASUAL' in context['leave_type'] else 'sl' if 'SICK' in context['leave_type'] else 'co'
            intent_hint = f"TASK: Query 'leavebalance' table. Filter by {leave_col} only. Use CAST({leave_col} AS DECIMAL(10,2)) in WHERE."
        elif "can" in q_lower and "apply" in q_lower:
            intent_hint = "TASK: Check if employee can apply for specified leave type by verifying balance > 0"
        else:
            intent_hint = "TASK: Query 'leavebalance' table. Return cl, sl, co values."
    elif intent == "leaves":
        if any(word in q_lower for word in ['latest', 'recent', 'last']):
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
    
    # CRITICAL: Add explicit warning against using example emails
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

CRITICAL DATE RANGE RULES:
1. When asked for "leave records between DATE1 and DATE2" OR "last month records" OR "records from DATE1 to DATE2":
   → Filter by leave start date: WHERE `from` BETWEEN 'DATE1' AND 'DATE2'
   
2. When asked "who is on leave on DATE" OR "members on leave on DATE" OR "who applied leave on DATE":
   → Find leaves that include that date: WHERE 'DATE' BETWEEN `from` AND `to`
   
3. When asked "who is on leave today":
   → Use: WHERE CURDATE() BETWEEN `from` AND `to`

Now generate MySQL query for:"""

    user_message = f"Question: {user_question}"
    
    # Llama 3.1 Instruct format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    return messages

# --------------------- SQL GENERATION WITH LLAMA 3.1 ---------------------

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
            print(f"\n⚠ Attempt {attempt + 1}/{max_attempts} failed: {error_msg[:150]}...")
            
            if attempt < max_attempts - 1:
                print("🔧 Repairing SQL...")
                current_sql = repair_sql_with_llm(current_sql, error_msg, tokenizer, model, schema, original_question, intent)
                print(f"✓ Repaired:\n{current_sql}\n")
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

def preformat_results(cols: list, rows: list, user_question: str) -> str:
    """
    Pre-format query results into bullet points to guarantee all rows are shown.
    """
    if not rows:
        return "No results found."
    
    formatted_lines = []
    q_lower = user_question.lower()
    
    # Detect what type of query this is
    is_balance_query = any(word in q_lower for word in ['balance', 'cl', 'sl', 'co'])
    is_leave_query = any(word in q_lower for word in ['leave', 'from', 'to', 'applied'])
    
    for row in rows:
        line_parts = []
        
        # Always start with name if available
        if 'empname' in [c.lower() for c in cols]:
            name_idx = [c.lower() for c in cols].index('empname')
            line_parts.append(row[name_idx])
        
        # Add email if available
        if 'empemail' in [c.lower() for c in cols]:
            email_idx = [c.lower() for c in cols].index('empemail')
            line_parts.append(f"({row[email_idx]})")
        
        # Add relevant data based on columns
        for i, col in enumerate(cols):
            col_lower = col.lower()
            
            # Skip already processed columns
            if col_lower in ['empname', 'empemail', 'id']:
                continue
            
            value = row[i]
            
            # Format based on column type
            if col_lower in ['cl', 'sl', 'co']:
                col_name = {'cl': 'Casual Leave', 'sl': 'Sick Leave', 'co': 'Comp Off'}[col_lower]
                line_parts.append(f"{col_name}: {value} days")
            elif col_lower == 'total_balance':
                line_parts.append(f"Total: {value} days")
            elif col_lower in ['from', 'to']:
                line_parts.append(f"{col}: {value}")
            elif col_lower == 'leavetype':
                line_parts.append(f"Type: {value}")
            elif col_lower == 'reason':
                # Truncate long reasons
                reason_text = str(value)[:50] + "..." if value and len(str(value)) > 50 else value
                line_parts.append(f"Reason: {reason_text}")
            elif col_lower == 'empph':
                line_parts.append(f"Phone: {value}")
            else:
                line_parts.append(f"{col}: {value}")
        
        # Join all parts
        formatted_lines.append("• " + " - ".join(str(p) for p in line_parts if p))
    
    return "\n".join(formatted_lines)

    
# --------------------- NATURAL LANGUAGE VERBALIZATION ---------------------

def verbalize_results(tokenizer, model, user_question: str, cols: list, rows: list, intent: str, page_size: int = 3, page_num: int = 1) -> str:
    """
    Convert SQL query results into natural language using Llama 3.1 8B with pagination
    """
    if not rows:
        return "No results found for your query."
    
    # Special handling for single-column, single-value results
    if len(cols) == 1 and len(rows) == 1:
        col_name = cols[0]
        value = rows[0][0]
        
        # Handle COUNT queries
        if 'COUNT' in col_name.upper():
            q_lower = user_question.lower()
            if 'how many' in q_lower or 'count' in q_lower:
                entity = 'employees'
                if 'leaves' in q_lower or 'applications' in q_lower:
                    entity = 'leave applications'
                return f"{value} {entity} match your criteria."
        
        # Extract email from question if present
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, user_question)
        
        if col_name.lower() in ['empph', 'phone', 'contact']:
            if email_match:
                return f"The phone number for {email_match.group()} is: {value}"
            else:
                return f"The phone number is: {value}"
        elif col_name.lower() in ['cl', 'sl', 'co']:
            return f"The {col_name.upper()} balance is: {value}"
    
    # Calculate pagination
    total_rows = len(rows)
    start_idx = (page_num - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    paginated_rows = rows[start_idx:end_idx]
    
    # Format data for the model
    data_summary = format_data_for_verbalization(cols, paginated_rows, page_num, start_idx)
    
    # With only 3-5 rows, we can use more tokens per row
    estimated_tokens = len(paginated_rows) * 150 + 300  # ✅ More generous estimate
    max_tokens = min(estimated_tokens, 1500)  # ✅ Lower ceiling since fewer rows
    
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
    
    # Clean up any artifacts from generation
    response = re.sub(r"^(Response:|Answer:|A:)\s*", "", response, flags=re.IGNORECASE).strip()
    
    # Add pagination info
    remaining = total_rows - end_idx
    if remaining > 0:
        response += f"\n\n📄 Showing results {start_idx + 1}-{end_idx} of {total_rows}. ({remaining} more remaining - type 'more' or 'next')"
    else:
        if total_rows > page_size:
            response += f"\n\n✓ Showing all {total_rows} results (page {page_num}/{(total_rows + page_size - 1) // page_size})."
    
    return response

    
def format_data_for_verbalization(cols: list, rows: list, page_num: int = 1, start_idx: int = 0) -> str:
    """
    Format query results into a readable string for the LLM
    """
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

# --------------------- PAGINATION STATE MANAGER ---------------------

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

# --------------------- MAIN ---------------------

def main():
    print("="*60)
    print("LLAMA 3.1 8B LEAVE CHATBOT v5.0 - WITH PAGINATION")
    print("="*60)
    
    tokenizer, model = load_model()
    print("✓ Llama 3.1 8B ready")
    
    intent_detector = IntentDetector()
    pagination_state = PaginationState()
    
    conn = get_db()
    print("✓ DB connected\n")
    print("Type 'quit' to exit")
    print("Type 'more' or 'next' to see more results\n")
    
    while True:
        try:
            q = input("💬 Your Question: ").strip()
            if q.lower() == "quit":
                break
            
            if not q:
                continue
            
            conn = ensure_connection(conn)
            # Check if user wants to see more results
            if q.lower() in ['more', 'next', 'show more', 'continue']:
                if not pagination_state.has_data():
                    print(" No previous query results to show more of. Please ask a new question.\n")
                    continue
                
                if not pagination_state.next_page():
                    print(" You've reached the end of the results.\n")
                    continue
                
                # Generate response for next page
                print(f"\n Loading page {pagination_state.current_page}...")
                nl_response = verbalize_results(
                    tokenizer, model,
                    pagination_state.last_question,
                    pagination_state.last_cols,
                    pagination_state.last_rows,
                    pagination_state.last_intent,
                    page_size=pagination_state.page_size,
                    page_num=pagination_state.current_page
                )
                
                print("\n" + "="*60)
                print("ANSWER:")
                print("="*60)
                print(nl_response)
                print("="*60 + "\n")
                continue
            
            # Reset pagination for new query
            pagination_state.reset()
            
            # Preprocess dates
            original_q = q
            print(q)
            q, date_context = preprocess_question(q)
            if date_context:
                print(f" {date_context}")
            if q != original_q:
                print(f" Preprocessed: {q}")
            
            # Detect intent
            intent, confidence = intent_detector.detect(q)
            print(f" Intent: {intent} (confidence: {confidence:.2f})")
            
            # Analyze context
            context = analyze_query_context(q)
            print(f" Query scope: {context['query_scope']}")
            if context['leave_type']:
                print(f" Leave type detected: {context['leave_type']}")
            
            # Get schema
            schema = get_schema_for_intent(intent)
            
            # Build Llama 3.1 prompt
            messages = build_llama_prompt(q, intent, context)
            
            # Generate SQL
            print(" Generating SQL...")
            raw_sql = generate_sql(tokenizer, model, messages)
            print(f"\n Generated SQL:\n{raw_sql}\n")
            
            # Validate and auto-repair
            print(" Validating SQL...")
            final_sql, is_valid, attempts = validate_and_repair_sql(
                conn, raw_sql, tokenizer, model, schema, q, intent, max_attempts=3
            )
            
            if is_valid:
                print(f" SQL validated (took {attempts} attempt(s))")
            else:
                print(f" Validation failed after {attempts} attempts, trying execution...")
            
            # Execute
            print("\nExecuting query...")
            cols, rows = run_query(conn, final_sql)
            
            # Store results for pagination
            pagination_state.set_results(final_sql, cols, rows, intent, original_q)
            
            # Display raw results (optional - can be commented out)
            if not rows:
                print(" No results found")
            else:
                print(f"\n Raw Results ({len(rows)} rows):")
                print("-" * 60)
                
                col_widths = [max(len(str(c)), max(len(str(row[i])) for row in rows[:5])) for i, c in enumerate(cols)]
                
                header = " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cols))
                print(f"  {header}")
                print(f"  {'-' * len(header)}")
                
                for row in rows[:5]:
                    row_str = " | ".join(str(v if v is not None else "NULL").ljust(col_widths[i]) for i, v in enumerate(row))
                    print(f"  {row_str}")
                
                if len(rows) > 5:
                    print(f"  ... and {len(rows)-5} more rows")
            
            # Generate natural language response with pagination
            print("\n💭 Generating natural language response...")
            nl_response = verbalize_results(
                tokenizer, model, original_q, cols, rows, intent,
                page_size=pagination_state.page_size,
                page_num=pagination_state.current_page
            )
            
            print("\n" + "="*60)
            print(" ANSWER:")
            print("="*60)
            print(nl_response)
            print("="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n⚠  Interrupted")
            break
        except Exception as e:
            print(f" Error: {e}")
    
    try:
        conn.close()
    except:
        pass
    print("\n Goodbye!")
    
if __name__ == "__main__":
    main()

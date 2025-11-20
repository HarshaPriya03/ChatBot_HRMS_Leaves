import mysql.connector
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    print("Loading Llama 3.1 8B Instruct...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Add pad token if not present
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            load_in_4bit=True,
        )
    except Exception:
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    mdl.eval()
    return tok, mdl

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
        print(" Loading intent detector...")
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
        print(" Intent detector ready")
    
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
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, question)
    if email_match:
        context['has_specific_email'] = True
        context['email'] = email_match.group()
        context['query_scope'] = 'specific_employee'
    
    # Detect leave type
    if 'sick' in q_lower:
        context['leave_type'] = 'SICK LEAVE'
    elif 'casual' in q_lower:
        context['leave_type'] = 'CASUAL LEAVE'
    elif 'comp' in q_lower:
        context['leave_type'] = 'COMP OFF'
    
    # Detect if asking about multiple/all employees
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
    # Fix NULLS FIRST/LAST
    sql = re.sub(r'\s+NULLS\s+(FIRST|LAST)', '', sql, flags=re.IGNORECASE)
    
    # Fix ILIKE to LIKE
    sql = re.sub(r'\bILIKE\b', 'LIKE', sql, flags=re.IGNORECASE)
    
    # Fix wrong column names
    sql = re.sub(r'\bltype\b', 'leavetype', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bemail\b(?![\w@])', 'empemail', sql, flags=re.IGNORECASE)
    
    # Fix DATEDIFF - MySQL takes 2 params, not 3
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
    
    # Fix CURRENT_DATE - INTERVAL to DATE_SUB
    sql = re.sub(
        r'\(?\s*CURRENT_DATE\s*-\s*INTERVAL\s+(\d+)\s+(DAY|WEEK|MONTH|YEAR)\s*\)?',
        r'DATE_SUB(CURDATE(), INTERVAL \1 \2)',
        sql,
        flags=re.IGNORECASE
    )
    
    # Replace CURRENT_DATE with CURDATE()
    sql = re.sub(r'\bCURRENT_DATE\(\)\b', 'CURDATE()', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bCURRENT_DATE\b', 'CURDATE()', sql, flags=re.IGNORECASE)
    
    # Ensure backticks for reserved words
    sql = re.sub(r'\bfrom\b(?![\s`])', '`from`', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bto\b(?![\s`])', '`to`', sql, flags=re.IGNORECASE)
    
    # FIX: Remove redundant date checks when using CURDATE() BETWEEN
    if 'CURDATE() BETWEEN' in sql.upper():
        # Remove redundant `from` <= date checks
        sql = re.sub(
            r"\s+AND\s+`from`\s*<=\s*['\"][\d-]+['\"]",
            "",
            sql,
            flags=re.IGNORECASE
        )
        # Remove redundant `to` >= date checks
        sql = re.sub(
            r"\s+AND\s+`to`\s*>=\s*['\"][\d-]+['\"]",
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
    
    # NEW FIX: Cast in WHERE clauses for numeric comparisons
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
""",
    
    "leaves": """
Table: leaves
Columns:
  - ID: PRIMARY KEY
  - empname: VARCHAR (employee name)
  - empemail: VARCHAR (employee email) - CRITICAL: Use 'empemail' NOT 'email'
  - leavetype: VARCHAR ('SICK LEAVE', 'CASUAL LEAVE', 'COMP OFF') - Use 'leavetype' NOT 'ltype'
  - applied: DATETIME (when leave was submitted)
  - `from`: DATE (leave start date) - MUST use backticks
  - `to`: DATE (leave end date) - MUST use backticks
  - desg: VARCHAR (designation)
  - reason: TEXT (leave reason)
  - empph: VARCHAR (employee phone)
  - work_location: VARCHAR

CRITICAL DATE QUERY RULES:
- For "on leave today" → WHERE CURDATE() BETWEEN `from` AND `to`
- For "on leave on specific date" → WHERE '2025-11-06' BETWEEN `from` AND `to`
- NEVER mix CURDATE() with specific date checks in same query
- For "latest/recent leave" → ORDER BY `from` DESC LIMIT 1
- For "latest [SICK/CASUAL/COMP] leave" → WHERE LOWER(leavetype) LIKE '%sick%' ORDER BY `from` DESC LIMIT 1
- For "when applied" → Use 'applied' column, ORDER BY applied DESC
- For duration: DATEDIFF(`to`, `from`) + 1
- For phone: SELECT empph
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
        if "can" in q_lower and "apply" in q_lower:
            intent_hint = "TASK: Check if employee can apply for specified leave type by verifying balance > 0"
        else:
            intent_hint = "TASK: Query 'leavebalance' table. Return cl, sl, co values."
    elif intent == "leaves":
        if any(word in q_lower for word in ['latest', 'recent', 'last']):
            # Add leave type filter if detected
            leave_type_filter = ""
            if context['leave_type']:
                leave_type_filter = f" AND LOWER(leavetype) LIKE '%{context['leave_type'].split()[0].lower()}%'"
            
            if context['has_specific_email']:
                intent_hint = f"TASK: Use 'leaves' table. Find most recent leave{leave_type_filter} for {context['email']}. ORDER BY `from` DESC LIMIT 1. Show `from`, `to`, leavetype, reason."
            else:
                intent_hint = f"TASK: Use 'leaves' table. Find most recent leave{leave_type_filter} for ALL employees. ORDER BY `from` DESC. Show empname, empemail, `from`, `to`, leavetype."
        elif any(word in q_lower for word in ['duration', 'how many days', 'how long']):
            intent_hint = "TASK: Calculate duration using DATEDIFF(`to`, `from`) + 1"
        elif "phone" in q_lower or "contact" in q_lower:
            intent_hint = "TASK: SELECT empph (phone number) from leaves table"
        elif any(word in q_lower for word in ['on leave', 'who is on leave', 'members on leave']):
            # Check if specific date or today
            date_pattern = r"on\s+'([\d-]+)'"
            date_match = re.search(date_pattern, user_question)
            if date_match:
                specific_date = date_match.group(1)
                intent_hint = f"TASK: Find employees on leave on {specific_date}. WHERE '{specific_date}' BETWEEN `from` AND `to`. DO NOT use CURDATE()."
            else:
                intent_hint = "TASK: Find employees currently on leave. WHERE CURDATE() BETWEEN `from` AND `to`"
        elif any(word in q_lower for word in ['top', 'most', 'maximum']):
            intent_hint = "TASK: GROUP BY empemail, COUNT leaves, ORDER BY count DESC, use LIMIT"
    
    # Build system prompt with examples
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

# Add these examples to the CRITICAL EXAMPLES section in build_llama_prompt()

CRITICAL EXAMPLES:
Q: "what is leave balance/ latest leave balance/ latest leavebalance of john@example.com"
A: SELECT cl, sl, co FROM leavebalance WHERE empemail = 'john@example.com';

Q: "who has the lowest casual leave balance"
A: SELECT empname, empemail, cl FROM leavebalance ORDER BY CAST(cl AS DECIMAL(10,2)) ASC LIMIT 1;

Q: "who has the highest casual leave balance"
A: SELECT empname, empemail, cl FROM leavebalance ORDER BY CAST(cl AS DECIMAL(10,2)) DESC LIMIT 1;

Q: "who has the lowest sick leave balance"
A: SELECT empname, empemail, sl FROM leavebalance ORDER BY CAST(sl AS DECIMAL(10,2)) ASC LIMIT 1;

Q: "what is the leave duration ofputsalaharshapriya@gmail.com in her last leave?" 
A: SELECT DATEDIFF(`to`, `from`) + 1 FROM leaves WHERE empemail = 'putsalaharshapriya@gmail.com' ORDER BY `from` DESC LIMIT 1;

Q: "top 5 employees with highest comp off balance"
A: SELECT empname, empemail, co FROM leavebalance ORDER BY CAST(co AS DECIMAL(10,2)) DESC LIMIT 5;

Q: "latest leave of john@example.com"
A: SELECT empname, `from`, `to`, leavetype, reason FROM leaves WHERE empemail = 'john@example.com' ORDER BY `from` DESC LIMIT 1;

Q: "latest sick leave of john@example.com"
A: SELECT empname, `from`, `to`, leavetype, reason FROM leaves WHERE empemail = 'john@example.com' AND LOWER(leavetype) LIKE '%sick%' ORDER BY `from` DESC LIMIT 1;

Q: "what is the latest leave applied"
A: SELECT empname, empemail, `from`, `to`, leavetype, applied FROM leaves ORDER BY applied DESC LIMIT 1;

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
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated_ids = out[0][inputs['input_ids'].shape[1]:]
    sql = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Extract SQL from response
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"^SQL:\s*|^A:\s*|^Answer:\s*", "", sql, flags=re.IGNORECASE).strip()
    
    # Take first complete SQL statement
    if ';' in sql:
        sql = sql.split(';')[0].strip() + ';'
    elif not sql.endswith(';'):
        sql += ';'
    
    # Apply quick fixes
    sql = quick_syntax_fix(sql)
    
    return sql

# --------------------- LLM REPAIR ---------------------

def repair_sql_with_llm(sql: str, error_msg: str, tokenizer, model, schema: str, original_question: str, intent: str) -> str:
    # Apply quick fixes first
    sql = quick_syntax_fix(sql)
    
    error_lower = error_msg.lower()
    
    # Identify specific errors
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
    
    fixed_sql = re.sub(r"^```sql\s*|\s*```$", "", fixed_sql, flags=re.IGNORECASE).strip()
    
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
            print(f"\n  Attempt {attempt + 1}/{max_attempts} failed: {error_msg[:150]}...")
            
            if attempt < max_attempts - 1:
                print(" Repairing SQL...")
                current_sql = repair_sql_with_llm(current_sql, error_msg, tokenizer, model, schema, original_question, intent)
                print(f" Repaired:\n{current_sql}\n")
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

# --------------------- MAIN ---------------------

def main():
    print("="*60)
    print("LLAMA 3.1 8B LEAVE CHATBOT v3.2 - FIXED")
    print("="*60)
    
    tokenizer, model = load_model()
    print(" Llama 3.1 8B ready")
    
    intent_detector = IntentDetector()
    
    conn = get_db()
    print(" DB connected\n")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            q = input(" Your Question: ").strip()
            if q.lower() == "quit":
                break
            
            if not q:
                continue
            
            conn = ensure_connection(conn)
            
            # Preprocess dates
            original_q = q
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
                print(f"  Leave type detected: {context['leave_type']}")
            
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
            print("\n Executing query...")
            cols, rows = run_query(conn, final_sql)
            
            # Display results
            if not rows:
                print(" No results found")
            else:
                print(f"\n Results ({len(rows)} rows):\n")
                
                col_widths = [max(len(str(c)), max(len(str(row[i])) for row in rows)) for i, c in enumerate(cols)]
                
                header = " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cols))
                print(f"  {header}")
                print(f"  {'-' * len(header)}")
                
                for row in rows[:20]:
                    row_str = " | ".join(str(v if v is not None else "NULL").ljust(col_widths[i]) for i, v in enumerate(row))
                    print(f"  {row_str}")
                
                if len(rows) > 20:
                    print(f"\n  ... and {len(rows)-20} more rows")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n Interrupted")
            break
        except Exception as e:
            print(f" Error: {e}")
    
    try:
        conn.close()
    except:
        pass
    print("\n Exit")

if __name__ == "__main__":
    main()

import mysql.connector
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import datetime
from sentence_transformers import SentenceTransformer, util

# --------------------- MODEL + DB ---------------------

MODEL_ID = "defog/sqlcoder-7b-2"

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
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
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
        question = question.replace("this month", f"after '{month_start}'")
    
    elif "yesterday" in q_lower:
        yesterday = today - datetime.timedelta(days=1)
        date_context = f"Date: {yesterday}"
        question = question.replace("yesterday", f"on '{yesterday}'")
    
    elif "last week" in q_lower:
        last_week_start = today - datetime.timedelta(days=today.weekday() + 7)
        last_week_end = last_week_start + datetime.timedelta(days=6)
        date_context = f"Date range: {last_week_start} to {last_week_end}"
        question = question.replace("last week", f"between '{last_week_start}' and '{last_week_end}'")
    
    return question, date_context

# --------------------- INTENT DETECTION ---------------------

class IntentDetector:
    def __init__(self):
        print("🔧 Loading intent detector...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.intent_examples = {
            "leavebalance": [
                "remaining leaves", "leave balance", "how many leaves left",
                "available leaves", "leaves available", "pending leave count",
                "current balance", "check balance", "balance remaining",
                "CL left", "SL available", "comp off balance",
                "can I take leave", "eligible for leave", "eligibility",
                "balance of", "leavebalance of", "what is balance",
                "can apply", "able to apply", "can take", "allowed to take"
            ],
            "leaves": [
                "applied leaves", "leave history", "leaves taken", 
                "reason for leave", "leave from date to date",
                "when did apply", "past leaves", "leave applications",
                "how many days applied", "total days taken",
                "leaves in January", "on leave today", "who is on leave",
                "leave between dates", "applied on", "show leaves",
                "latest leave", "last leave", "recent leave", "when applied",
                "who applied", "which employee", "person applied", "members on leave",
                "phone number", "empph", "contact", "records", "list of",
                "from date", "to date", "leave period", "leave duration"
            ]
        }
        
        self.intent_embeds = {
            k: self.model.encode(v, convert_to_tensor=True) 
            for k, v in self.intent_examples.items()
        }
        print("✅ Intent detector ready")
    
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
            elif any(word in q_lower for word in ['applied', 'history', 'latest', 'reason', 'when', 'days', 'who', 'phone', 'empph', 'contact', 'from', 'to']):
                return "leaves", 0.6
            return "unknown", best_score
        
        return best_intent, best_score

# --------------------- CONTEXT ANALYSIS ---------------------

def analyze_query_context(question: str) -> dict:
    """Analyzes the question to extract context dynamically"""
    q_lower = question.lower()
    context = {
        'has_specific_email': False,
        'email': None,
        'is_general_query': False,
        'query_scope': 'unknown'
    }
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, question)
    if email_match:
        context['has_specific_email'] = True
        context['email'] = email_match.group()
        context['query_scope'] = 'specific_employee'
        return context
    
    # Detect if asking about multiple/all employees
    question_indicators = ['who', 'which', 'what', 'how many', 'list', 'show']
    person_plurals = ['employees', 'members', 'people', 'persons', 'staff']
    quantifiers = ['all', 'any', 'every', 'everyone', 'anyone']
    
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
    """Apply immediate regex-based fixes for common issues BEFORE validation"""
    
    # Fix NULLS FIRST/LAST
    sql = re.sub(r'\s+NULLS\s+(FIRST|LAST)', '', sql, flags=re.IGNORECASE)
    
    # Fix ILIKE to LIKE
    sql = re.sub(r'\bILIKE\b', 'LIKE', sql, flags=re.IGNORECASE)
    
    # Fix wrong column names
    sql = re.sub(r'\bltype\b', 'leavetype', sql, flags=re.IGNORECASE)
    
    # Fix DATEDIFF - MySQL takes 2 params, not 3
    sql = re.sub(
        r"DATEDIFF\s*\(\s*['\"]day['\"]\s*,\s*([^,]+)\s*,\s*([^)]+)\)",
        r"DATEDIFF(\2, \1)",
        sql,
        flags=re.IGNORECASE
    )
    
    # Fix INTERVAL syntax: INTERVAL 'X unit' -> INTERVAL X unit
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
    
    return sql

# --------------------- SCHEMA TEMPLATES ---------------------

SCHEMA_TEMPLATES = {
    "leavebalance": """
Table: leavebalance
Columns:
  - id: PRIMARY KEY
  - empname: VARCHAR (employee name)
  - empemail: VARCHAR (employee email) - USE THIS, NOT 'email'
  - cl: VARCHAR (Casual Leave balance - CAST to DECIMAL for math)
  - sl: VARCHAR (Sick Leave balance - CAST to DECIMAL for math)
  - co: VARCHAR (Comp Off balance - CAST to DECIMAL for math)
  - lastupdate: DATETIME
  - icl, isl, ico, iupdate: initial values

CRITICAL RULES:
- For questions "can X apply for leave" or "leave balance": Query leavebalance table
- Column name is 'empemail' NOT 'email'
- To check if someone can take CL: SELECT cl FROM leavebalance WHERE empemail='...'
- To check if someone can take SL: SELECT sl FROM leavebalance WHERE empemail='...'
- To check if someone can take CO: SELECT co FROM leavebalance WHERE empemail='...'
- cl/sl/co are VARCHAR - CAST to DECIMAL: CAST(cl AS DECIMAL(10,2)) > 0
""",
    
    "leaves": """
Table: leaves
Columns:
  - ID: PRIMARY KEY
  - empname: VARCHAR (employee name)
  - empemail: VARCHAR (employee email) - USE THIS, NOT 'email'
  - leavetype: VARCHAR (values: 'Casual Leave', 'Sick Leave', 'Comp Off') - USE THIS, NOT 'ltype'
  - applied: DATETIME (when the leave was applied/submitted)
  - `from`: DATE (leave start date) - MUST use backticks
  - `to`: DATE (leave end date) - MUST use backticks
  - desg: VARCHAR (designation)
  - reason: TEXT (leave reason)
  - empph: VARCHAR (employee phone)
  - work_location: VARCHAR

CRITICAL RULES:
- For "latest leave", "recent leave", "when took leave": Use `from` and `to` dates, NOT 'applied'
- Column is 'leavetype' NOT 'ltype'
- Use UPPER() or LOWER() for case-insensitive matching: LOWER(leavetype) = 'casual leave'
- For latest leave: ORDER BY `from` DESC or `to` DESC
- For applied date: Use 'applied' column
- For leave period/duration: Use `from` and `to` columns
- Backticks required: `from`, `to`
- For phone: SELECT empph FROM leaves WHERE empemail='...'
- MySQL DATEDIFF: DATEDIFF(`to`, `from`) + 1 for total days
"""
}

def get_schema_for_intent(intent: str) -> str:
    if intent == "leavebalance":
        return SCHEMA_TEMPLATES["leavebalance"]
    elif intent == "leaves":
        return SCHEMA_TEMPLATES["leaves"]
    else:
        return SCHEMA_TEMPLATES["leavebalance"] + "\n" + SCHEMA_TEMPLATES["leaves"]

# --------------------- PROMPT BUILDING ---------------------

def build_prompt(user_question: str, intent: str, context: dict) -> str:
    schema = get_schema_for_intent(intent)
    
    context_hint = ""
    if context['has_specific_email']:
        context_hint = f"\nREQUIRED: WHERE empemail = '{context['email']}'\n"
    elif context['is_general_query']:
        context_hint = "\nREQUIRED: Query ALL employees. NO email filter. Include empemail in SELECT.\n"
    
    # Add intent-specific hints
    intent_hint = ""
    q_lower = user_question.lower()
    
    if intent == "leavebalance":
        intent_hint = "\nFOR THIS QUERY: Use 'leavebalance' table to check cl/sl/co balance. Return balance value.\n"
    elif intent == "leaves":
        if any(word in q_lower for word in ['latest', 'recent', 'last', 'when took', 'from', 'to']):
            intent_hint = "\nFOR THIS QUERY: Use 'leaves' table. Show `from` and `to` dates (leave period), NOT 'applied' date. ORDER BY `from` DESC.\n"
        else:
            intent_hint = "\nFOR THIS QUERY: Use 'leaves' table with column 'leavetype' (NOT 'ltype'). Use LOWER(leavetype) for matching.\n"
    
    prompt = f"""### Task
Generate MySQL/MariaDB query. ONLY use tables 'leavebalance' and 'leaves'.

### Schema
{schema}

### MySQL/MariaDB Syntax (CRITICAL)
- Column is 'leavetype' NOT 'ltype'
- Column is 'empemail' NOT 'email'
- For case-insensitive: LOWER(leavetype) = 'casual leave'
- For latest leave: ORDER BY `from` DESC (use leave start date, not applied date)
- DATEDIFF(end, start) - 2 params only
- Backticks for reserved words: `from`, `to`
- CURDATE() not CURRENT_DATE
- NO NULLS FIRST/LAST
- DATE_SUB(CURDATE(), INTERVAL X DAY/MONTH/YEAR)
{context_hint}{intent_hint}

### Question
{user_question}

### SQL:
"""
    return prompt

# --------------------- LLM REPAIR (IMPROVED) ---------------------

def repair_sql_with_llm(sql: str, error_msg: str, tokenizer, model, schema: str, original_question: str, intent: str) -> str:
    """Improved repair with error-specific fixes"""
    
    # First apply quick fixes
    sql = quick_syntax_fix(sql)
    
    # Check if error still exists after quick fix
    error_lower = error_msg.lower()
    
    # Build targeted repair hints
    if "'ltype'" in error_msg or "ltype" in error_lower:
        repair_hint = "ERROR: Column 'ltype' does not exist. The correct column name is 'leavetype'. Replace ltype with leavetype."
        sql = re.sub(r'\bltype\b', 'leavetype', sql, flags=re.IGNORECASE)
    elif "doesn't exist" in error_lower and "table" in error_lower:
        repair_hint = "ERROR: Wrong table. ONLY use 'leavebalance' or 'leaves' tables."
    elif "'email'" in error_msg or (("column" in error_lower or "unknown" in error_lower) and "email" in error_lower):
        repair_hint = "ERROR: Column 'email' doesn't exist. Use 'empemail' instead."
        sql = re.sub(r'\bemail\b(?!\w)', 'empemail', sql, flags=re.IGNORECASE)
    elif "datediff" in error_lower:
        repair_hint = "ERROR: DATEDIFF takes 2 params in MySQL: DATEDIFF(end_date, start_date)"
    elif "reserved" in error_lower or "syntax" in error_lower:
        repair_hint = "ERROR: 'from' and 'to' are reserved keywords. Use backticks: `from`, `to`"
        sql = re.sub(r'\bfrom\b(?![`\w])', '`from`', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bto\b(?![`\w])', '`to`', sql, flags=re.IGNORECASE)
    else:
        repair_hint = f"ERROR: {error_msg[:150]}"
    
    repair_prompt = f"""Fix this MySQL query for the 'ems' database.

{repair_hint}

Schema: {schema}

Original Question: {original_question}

Broken SQL:
{sql}

Output ONLY the corrected SQL with no explanations or comments:
"""
    
    inputs = tokenizer(repair_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    fixed_sql = text.replace(repair_prompt, "").strip()
    fixed_sql = re.sub(r"^```sql\s*|\s*```$", "", fixed_sql, flags=re.IGNORECASE).strip()
    
    if not fixed_sql.endswith(';'):
        fixed_sql = fixed_sql.rstrip() + ';'
    
    # Apply quick fixes to repaired SQL
    fixed_sql = quick_syntax_fix(fixed_sql)
    
    return fixed_sql

# --------------------- SQL GENERATION ---------------------

def generate_sql(tokenizer, model, prompt: str, max_new_tokens: int = 350) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    
    if "### SQL:" in text:
        sql = text.split("### SQL:")[-1].strip()
    else:
        sql = text.replace(prompt, "").strip()
    
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
    
    if ';' in sql:
        sql = sql.split(';')[0].strip() + ';'
    elif not sql.endswith(';'):
        sql += ';'
    
    if "__SORRY__" in sql.upper():
        return "__SORRY__"
    
    # Apply quick fixes immediately
    sql = quick_syntax_fix(sql)
    
    return sql

def validate_and_repair_sql(conn, sql: str, tokenizer, model, schema: str, original_question: str, intent: str, max_attempts: int = 3) -> tuple:
    """Validates SQL and uses LLM to repair if invalid"""
    
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
            print(f"\n⚠️  Attempt {attempt + 1}/{max_attempts} failed: {error_msg[:150]}...")
            
            if attempt < max_attempts - 1:
                print("🔧 Repairing SQL...")
                current_sql = repair_sql_with_llm(current_sql, error_msg, tokenizer, model, schema, original_question, intent)
                print(f"🔄 Repaired:\n{current_sql}\n")
            else:
                return current_sql, False, attempt + 1
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n⚠️  Attempt {attempt + 1}/{max_attempts} failed: {error_msg[:150]}...")
            
            if attempt < max_attempts - 1:
                print("🔧 Repairing SQL...")
                current_sql = repair_sql_with_llm(current_sql, error_msg, tokenizer, model, schema, original_question, intent)
                print(f"🔄 Repaired:\n{current_sql}\n")
            else:
                return current_sql, False, attempt + 1
        finally:
            if cursor:
                cursor.close()
    
    return current_sql, False, max_attempts

def run_query(conn, sql: str):
    if sql.strip() == "__SORRY__":
        return "__SORRY__", []
    
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
    print("INTELLIGENT LEAVE CHATBOT v2.1 - FIXED")
    print("="*60)
    
    print("\n🔧 Loading SQL model...")
    tokenizer, model = load_model()
    print("✅ SQL model ready")
    
    intent_detector = IntentDetector()
    
    conn = get_db()
    print("✅ DB connected\n")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            q = input("💬 Your Question: ").strip()
            if q.lower() == "quit":
                break
            
            if not q:
                continue
            
            conn = ensure_connection(conn)
            
            # Preprocess dates
            original_q = q
            q, date_context = preprocess_question(q)
            if date_context:
                print(f"📅 {date_context}")
            if q != original_q:
                print(f"🔄 Preprocessed: {q}")
            
            # Detect intent
            intent, confidence = intent_detector.detect(q)
            print(f"🎯 Intent: {intent} (confidence: {confidence:.2f})")
            
            # Analyze context
            context = analyze_query_context(q)
            print(f"🔍 Query scope: {context['query_scope']}")
            
            # Get schema
            schema = get_schema_for_intent(intent)
            
            # Build prompt
            prompt = build_prompt(q, intent, context)
            
            # Generate SQL
            print("⏳ Generating SQL...")
            raw_sql = generate_sql(tokenizer, model, prompt)
            print(f"\n📝 Generated SQL:\n{raw_sql}\n")
            
            if raw_sql == "__SORRY__":
                print("❌ Cannot generate query")
                continue
            
            # Validate and auto-repair with LLM
            print("🔍 Validating SQL...")
            final_sql, is_valid, attempts = validate_and_repair_sql(
                conn, raw_sql, tokenizer, model, schema, q, intent, max_attempts=3
            )
            
            if is_valid:
                print(f"✅ SQL validated (took {attempts} attempt(s))")
            else:
                print(f"⚠️ Validation failed after {attempts} attempts, trying execution...")
            
            # Execute
            print("\n🔄 Executing query...")
            cols, rows = run_query(conn, final_sql)
            
            if isinstance(cols, str) and cols == "__SORRY__":
                print("❌ Execution failed")
                continue
            
            # Display
            if not rows:
                print("📭 No results found")
            else:
                print(f"\n📊 Results ({len(rows)} rows):\n")
                
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
            print("\n\n👋 Interrupted")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    try:
        conn.close()
    except:
        pass
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
